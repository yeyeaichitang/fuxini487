"""Inject a reversible LoRA adapter INTO FuXi-2.1's original exported FX graph.

This operates on the GraphModule returned from torch.export.load(...).module().
It is NOT a standalone replacement for the original FuXi weather backbone.
No model training and no PV data alignment are performed here.
"""
from __future__ import annotations

from pathlib import Path
import torch
from torch import nn

WQ_NAME = 'model.decoder.layers.29.attn.wq.weight'
NODE_NAME = 'linear_235'
A_NAME = 'fuxi_lora29_wq_A'
B_NAME = 'fuxi_lora29_wq_B'


def inject_original_fuxi_lora(model: torch.fx.GraphModule, rank: int = 4, alpha: float = 8.0):
    """Patch original full FX graph in memory; returns same GraphModule instance.

    At zero-initialized B, the original output must be unchanged.
    Only A/B are trainable. Call before saving/loading adapter tensors.
    """
    if rank <= 0 or alpha <= 0:
        raise ValueError('rank and alpha must be positive')
    if hasattr(model, A_NAME) or hasattr(model, B_NAME):
        raise RuntimeError('LoRA already injected into this model')

    nodes = [n for n in model.graph.nodes if n.name == NODE_NAME]
    if len(nodes) != 1 or nodes[0].target != torch.ops.aten.linear.default:
        raise RuntimeError(f'Cannot find expected original FuXi op {NODE_NAME}')
    target = nodes[0]
    if len(target.args) < 2:
        raise RuntimeError('Expected linear(x, weight[, bias])')
    w_node = target.args[1]
    if not isinstance(w_node, torch.fx.Node) or w_node.op != 'get_attr' or w_node.target != WQ_NAME:
        # Exported model can expose named parameter via another graph form.
        # Refuse to risk modifying an incorrect layer.
        raise RuntimeError(f'Unexpected weight node for {NODE_NAME}: {w_node}')

    params = dict(model.named_parameters())
    if WQ_NAME not in params:
        raise RuntimeError('Original FuXi wq weight parameter not found')
    base_weight = params[WQ_NAME]
    if tuple(base_weight.shape) != (1560, 1536):
        raise RuntimeError(f'Unexpected original wq shape {tuple(base_weight.shape)}')

    for param in model.parameters():
        param.requires_grad_(False)

    # Stable float32 LoRA math. Original FuXi half precision path remains intact.
    device = base_weight.device
    A = nn.Parameter(torch.randn(rank, 1536, dtype=torch.float32, device=device) * 0.01)
    B = nn.Parameter(torch.zeros(1560, rank, dtype=torch.float32, device=device))
    model.register_parameter(A_NAME, A)
    model.register_parameter(B_NAME, B)
    graph = model.graph
    previous_users = list(target.users)
    with graph.inserting_before(target):
        attr_a = graph.get_attr(A_NAME)
        attr_b = graph.get_attr(B_NAME)
        float_x = graph.call_function(torch.ops.aten._to_copy.default, (target.args[0],), {'dtype': torch.float32})
        low = graph.call_function(torch.ops.aten.linear.default, (float_x, attr_a))
        delta = graph.call_function(torch.ops.aten.linear.default, (low, attr_b))
        delta = graph.call_function(torch.ops.aten.mul.Scalar, (delta, float(alpha) / rank))
        delta = graph.call_function(torch.ops.aten._to_copy.default, (delta,), {'dtype': base_weight.dtype})
    with graph.inserting_after(target):
        combined = graph.call_function(torch.ops.aten.add.Tensor, (target, delta))
    for user in previous_users:
        user.replace_input_with(target, combined)
    graph.lint()
    model.recompile()
    return model


def save_lora(model, destination: str | Path, *, rank: int, alpha: float):
    """Save ONLY original-FuXi graph adapter, not the huge pretrained model."""
    dst = Path(destination)
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'format': 'fuxi2.1_original_graph_lora_v1',
        'target_node': NODE_NAME,
        'target_weight': WQ_NAME,
        'rank': int(rank),
        'alpha': float(alpha),
        'A': getattr(model, A_NAME).detach().cpu().clone(),
        'B': getattr(model, B_NAME).detach().cpu().clone(),
        'training_completed': False,
        'supervision': None,
    }, dst)
    return dst


def load_lora(model, source: str | Path, *, rank: int, alpha: float):
    """Load only adapters saved for THIS original graph; no bridge checkpoints."""
    obj = torch.load(source, map_location='cpu', weights_only=True)
    if obj.get('format') != 'fuxi2.1_original_graph_lora_v1':
        raise RuntimeError('Not an original-graph LoRA adapter. Bridge checkpoints are NOT compatible.')
    if obj.get('target_node') != NODE_NAME or obj.get('target_weight') != WQ_NAME:
        raise RuntimeError('Adapter target mismatch')
    if obj.get('rank') != rank or float(obj.get('alpha')) != alpha:
        raise RuntimeError('Adapter rank/alpha mismatch')
    for key, name, shape in [('A', A_NAME, (rank, 1536)), ('B', B_NAME, (1560, rank))]:
        val = obj[key]
        if tuple(val.shape) != shape or not torch.isfinite(val).all():
            raise RuntimeError(f'Adapter tensor {key} is invalid')
        p = getattr(model, name)
        with torch.no_grad():
            p.copy_(val.to(p.device, dtype=p.dtype))
    return obj
