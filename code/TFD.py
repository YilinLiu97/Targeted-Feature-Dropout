import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TFD(nn.Module):

    def __init__(self, channels, test_state=False, M_ratio_range=[0.85,0.9], drop_mean_std=[0.2,0.05]):
       super(TFD, self).__init__()

       assert M_ratio_range[1] > M_ratio_range[0]

       self.test_state = test_state
       self.channels = channels
       self.avg_pool = nn.AdaptiveAvgPool3d(1)
       self.M_ratio_range = M_ratio_range
       self.drop_mean, self.drop_std = drop_mean_std[0], drop_mean_std[1]


    def forward(self, inputs):
       bs, chns, _,_,_ = inputs.size()

       scores = self.avg_pool(inputs).to(dtype=torch.float).cuda()  # size: BxCx1x1x1

       # M -- with variable ratio
       torch.set_printoptions(precision=1)
       M_ratio = self.M_ratio_range[0] + np.random.rand(1)*(self.M_ratio_range[1] - self.M_ratio_range[0])
       M = int(np.ceil(self.channels * M_ratio))

       # Get M largest elements, without replacement
       keys_to_choose = torch.multinomial(scores.cpu(), M)

       # Mask
       mask = torch.zeros(scores.size()).cuda()
       mask[torch.arange(mask.size(0)).unsqueeze(1), keys_to_choose] = 1

       # RNG -- with variable rate
       torch.set_printoptions(precision=1)

       RNG_drop = np.random.normal(loc = self.drop_mean, scale = self.drop_std, size =1)
       if RNG_drop < 0:
          RNG_drop = np.array(0)

       RNG_drop = torch.from_numpy(RNG_drop).to(dtype=torch.float)
       RNG_mask = (torch.FloatTensor(mask.size()).uniform_() > RNG_drop).float().cuda()
       dropped_mask = mask*RNG_mask

       if self.test_state:
          out = inputs
       else:
          out = inputs*dropped_mask

       return out.cuda()
