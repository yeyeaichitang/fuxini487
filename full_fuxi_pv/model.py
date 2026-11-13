"""Full original FuXi graph + internal LoRA + station weather/history fusion head.

The native FuXi forecast is six-hourly. The learned PV head predicts +15 minutes
using that forecast context and a 16-point measured station history.
"""
from pathlib import Path
import sys
import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'fuxi-2.1'))
from fuxi_lora import inject_original_fuxi_lora, A_NAME, B_NAME, WQ_NAME
from variables import C85_CHANNEL_NAMES

CONTEXT_CHANNELS = ['t2m', 'd2m', 'msl', 'ws10m', 'ws100m', 'ssrd', 'tcc', 'tcw']


class FullFuXiPV(nn.Module):
    def __init__(self, model_path, rank=4, alpha=8.0, device='cuda',weight_cache=None):
        super().__init__()
        self.rank, self.alpha = rank, alpha
        self.device = torch.device(device)
        if self.device.type != 'cuda':
            raise ValueError('The original exported FuXi graph requires CUDA')
        with torch.cuda.device(self.device):
            if weight_cache is None:
                exported = torch.export.load(model_path)
            else:
                from offload import load_export_mapped
                exported = load_export_mapped(model_path,weight_cache)
            self.fuxi = exported.module()
            del exported
        inject_original_fuxi_lora(self.fuxi, rank, alpha)
        for name in (A_NAME,B_NAME):
            parameter=getattr(self.fuxi,name)
            parameter.data=parameter.data.to(self.device)
        self.head = nn.Sequential(nn.Linear(16*5+len(CONTEXT_CHANNELS),64), nn.GELU(),
                                  nn.Linear(64,1), nn.Sigmoid()).to(self.device)
        self.channels = [C85_CHANNEL_NAMES.index(name) for name in CONTEXT_CHANNELS]
        self.last_context = None
        self.interpreter=None
        if weight_cache is not None:
            from offload import StreamingWeightInterpreter
            self.interpreter=StreamingWeightInterpreter(self.fuxi,self.device)

    def forward(self, state, temporal, history, grid_index):
        if tuple(state.shape) != (1,2,85,721,1440):
            raise ValueError('FuXi input must be [1,2,85,721,1440]')
        if tuple(history.shape) != (1,16,5):
            raise ValueError('Station history must be [1,16,5]')
        lat_index,lon_index = grid_index
        if not (0 <= lat_index < 721 and 0 <= lon_index < 1440):
            raise ValueError('Invalid station grid index')
        output = self.weather_forward(state,temporal)
        # No detach: the PV loss must propagate into LoRA inside the full graph.
        context = output[:, -1, self.channels, lat_index, lon_index].float()
        if not torch.isfinite(context).all():
            raise ValueError('Nonfinite FuXi station forecast context')
        self.last_context = context.detach().cpu()
        return self.head(torch.cat([history.float().flatten(1),context],dim=1)).squeeze(-1)

    def weather_forward(self,state,temporal):
        runner=self.interpreter.run if self.interpreter is not None else self.fuxi
        output=runner(state,temporal['step'],temporal['hour'],temporal['doy'])
        if self.interpreter is not None:
            self.interpreter.env.clear()
        return output

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def adapter_state(self):
        return {'A':getattr(self.fuxi,A_NAME).detach().cpu().clone(),
                'B':getattr(self.fuxi,B_NAME).detach().cpu().clone(),
                'head':{k:v.detach().cpu().clone() for k,v in self.head.state_dict().items()}}

    def load_adapter_state(self,state):
        with torch.no_grad():
            getattr(self.fuxi,A_NAME).copy_(state['A'].to(self.device))
            getattr(self.fuxi,B_NAME).copy_(state['B'].to(self.device))
        self.head.load_state_dict(state['head'])


def temporal_features(utc_time, device):
    tod=(utc_time.hour*60+utc_time.minute)/1440
    doy=min(365,utc_time.timetuple().tm_yday)/365
    return {k:torch.tensor([v],dtype=torch.float16,device=device)
            for k,v in {'step':0,'hour':tod,'doy':doy}.items()}


def state_from_array(array,device):
    values = np.asarray(array,dtype=np.float16)
    if values.shape != (2,85,721,1440) or np.isinf(values).any():
        raise ValueError('Invalid global weather input')
    # Official sample has masked SST over land. The original graph begins with
    # aten.nan_to_num; preserve this documented mask rather than filling sensors.
    for channel in range(85):
        if channel != C85_CHANNEL_NAMES.index('sst') and np.isnan(values[:,channel]).any():
            raise ValueError(f'Missing non-SST weather channel: {C85_CHANNEL_NAMES[channel]}')
    return torch.from_numpy(values[None]).to(device)
