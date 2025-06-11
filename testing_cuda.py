
import torch
res = torch.fft.rfft(torch.randn(1000).cuda())
print(res)
