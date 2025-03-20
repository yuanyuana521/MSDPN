import torch
from thop import profile

from models.MSDPN import build_net

model = build_net()
input1 = torch.randn(1, 3, 256, 256)
flops, params = profile(model, inputs=(input1, ))
print('FLOPs = ' + str(flops/1000**3) + 'G')
print('Params = ' + str(params/1000**2) + 'M')

