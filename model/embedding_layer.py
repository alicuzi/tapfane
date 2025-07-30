import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

import math

class DSW_embedding(nn.Module):
    def __init__(self, seg_len, d_model):
        super(DSW_embedding, self).__init__()
        self.seg_len = seg_len

        self.linear = nn.Linear(seg_len, d_model)

    def forward(self, x):
        batch, ts_len, ts_dim = x.shape
        # time series length is subdivided as seg_num * seg_len = ts_len
        x_segment = rearrange(x, 'b (seg_num seg_len) d -> (b d seg_num) seg_len', seg_len = self.seg_len)
        x_embed = self.linear(x_segment)
        x_embed = rearrange(x_embed, '(b d seg_num) d_model -> b d seg_num d_model', b = batch, d = ts_dim)
        
        return x_embed


class GDSW_embedding(nn.Module):
    def __init__(self, seg_len, d_model, d_spacefeatures = 3):
        super(GDSW_embedding, self).__init__()
        self.seg_len = seg_len
        self.geoemb = nn.Linear(d_spacefeatures, d_model)
        self.linear = nn.Linear(seg_len, d_model)

    def forward(self, x, x_geo):
        batch, ts_len, ts_dim = x.shape
        # time series length is subdivided as seg_num * seg_len = ts_len
        x_segment = rearrange(x, 'b (seg_num seg_len) d -> (b d seg_num) seg_len', seg_len = self.seg_len)
        # x_segment consists of batches of length seg_len for each dimension
        x_embed = self.linear(x_segment) + self.geoemb(x_geo[0]) # add contribution from cooridinate embedding
        x_embed = rearrange(x_embed, '(b d seg_num) d_model -> b d seg_num d_model', b = batch, d = ts_dim)
        # print(x_segment.shape, x_embed.shape)
        
        return x_embed