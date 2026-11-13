"""Disk-backed saved tensors for the late-layer full-graph backward pass."""
from contextlib import contextmanager
from pathlib import Path
import tempfile
import torch
from torch.utils._pytree import tree_map


class StreamingWeightInterpreter(torch.fx.Interpreter):
    """Execute every original FX node; stream disk-backed frozen weights to CUDA."""
    def __init__(self,module,device):
        super().__init__(module,garbage_collect_values=True)
        self.device=device

    def call_function(self,target,args,kwargs):
        def promote(value):
            if isinstance(value,torch.Tensor) and value.device.type=='cpu':
                return value.to(self.device)
            return value
        return target(*tree_map(promote,args),**tree_map(promote,kwargs))


def offload_frozen_weights(module,directory):
    """Keep immutable large parameters in mmap files, not resident GPU allocations."""
    directory=Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    total=0
    entries=list(module.named_parameters())+list(module.named_buffers())
    for index,(name,tensor) in enumerate(entries):
        if tensor.requires_grad or tensor.numel()*tensor.element_size()<1024*1024:
            continue
        path=directory/f'{index:05d}.pt'
        if not path.exists():
            torch.save(tensor.detach().cpu(),path)
        mapped=torch.load(path,map_location='cpu',mmap=True,weights_only=True)
        if mapped.shape!=tensor.shape or mapped.dtype!=tensor.dtype:
            raise ValueError('Weight cache incompatible')
        total+=tensor.numel()*tensor.element_size()
        if isinstance(tensor,torch.nn.Parameter):
            tensor.data=mapped
        else:
            parent,_,leaf=name.rpartition('.')
            setattr(module.get_submodule(parent) if parent else module,leaf,mapped)
    torch.cuda.empty_cache()
    print(f'Frozen weights memory-mapped: {total/1024**3:.2f} GiB',flush=True)


def load_export_mapped(model_path,directory):
    """Deserialize raw PT2 storages straight into mmap files to avoid load peaks.

    Scoped hook for the installed PyTorch 2.11 archive format. Restored even on
    failure; no installed package is edited. The interpreter supplies CUDA weights.
    """
    import torch.export.pt2_archive._package as package
    directory=Path(directory)
    directory.mkdir(parents=True,exist_ok=False)
    original=package._create_flat_tensor_from_bytes
    count=0
    def create(raw,meta):
        nonlocal count
        dtype=package.deserialize_scalar_type(meta.dtype)
        if not raw:
            return torch.zeros(package.deserialize_size(meta.sizes),dtype=dtype)
        path=directory/f'{count:05d}.bin'
        count+=1
        path.write_bytes(raw)
        width=torch.empty((),dtype=dtype).element_size()
        if len(raw)%width:
            raise ValueError('Invalid tensor storage length')
        # Shared file backing avoids Windows committing a private copy of all
        # 3.6 GB of immutable storages. These cache files are not the source PT2.
        result=torch.from_file(str(path),shared=True,size=len(raw)//width,dtype=dtype)
        result.requires_grad_(meta.requires_grad)
        return result
    package._create_flat_tensor_from_bytes=create
    try:
        exported=torch.export.load(model_path)
    finally:
        package._create_flat_tensor_from_bytes=original
    print(f'Original PT2 deserialized with {count} disk-backed storages',flush=True)
    return exported


@contextmanager
def saved_tensors_on_disk(parent):
    parent=Path(parent)
    parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='autograd_',dir=parent) as temporary:
        count=0
        total=0
        restored=0
        def pack(tensor):
            nonlocal count,total
            path=Path(temporary)/f'{count:06d}.bin'
            count+=1
            size=tensor.numel()*tensor.element_size()
            total+=size
            if total>20*1024**3:
                raise RuntimeError('Saved activations exceed 20 GiB disk budget')
            host=tensor.detach().cpu().contiguous()
            host.reshape(-1).view(torch.uint8).numpy().tofile(path)
            if count%20==0:
                print(f'Activation offload: {count} tensors, {total/1024**3:.2f} GiB',flush=True)
            return str(path),tensor.device,tensor.dtype,tuple(tensor.shape),tensor.numel()
        def unpack(item):
            nonlocal restored
            path,device,dtype,shape,numel=item
            restored+=1
            tensor=torch.from_file(path,shared=True,size=numel,dtype=dtype).reshape(shape)
            torch.cuda.empty_cache()
            result=tensor.to(device)
            if restored%10==0:
                print(f'Backward tensors restored: {restored}; CUDA allocated {torch.cuda.memory_allocated()/1024**3:.2f} GiB',flush=True)
            return result
        with torch.autograd.graph.saved_tensors_hooks(pack,unpack):
            yield
