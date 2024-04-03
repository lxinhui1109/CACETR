# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Various positional encodings for the transformer.
"""
import math
import torch
from torch import nn

from util.back import NestedTensor

#正余弦编码 就是attention那篇论文里的公式
class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors#x大小是[batch c h w] x有padding操作 [2,2048,26,25]
        mask = tensor_list.mask#标记一下x里的特征是实际的还是padding的 false是实际 true是padding  大小[2,26,25]
        assert mask is not None
        not_mask = ~mask #上面的mask取反 true是实际有值的 false是padding出来的，没有意义了
        y_embed = not_mask.cumsum(1, dtype=torch.float32)#行方向累加[2,26,25]
        x_embed = not_mask.cumsum(2, dtype=torch.float32)#列方向累加[2,26,25]
        if self.normalize:#归一化操作
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale
        #具体的公式 偶数维度和奇数维度的方法不一样
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)#128维
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)#128维

        pos_x = x_embed[:, :, :, None] / dim_t #embedding后行的位置[2,26,25,128]
        pos_y = y_embed[:, :, :, None] / dim_t #embedding后列的位置[2,26,25,128]
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)#[2,26,25,128]
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)#[2,26,25,128]
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)#[2,256,26,25]
        #cat操作 前128代表embeddingX 后128代表embeddingY 得到了26*25每一个位置的实际编码是什么
        return pos

#可学习的位置编码
class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, num_pos_feats=256):
        super().__init__()
        self.row_embed = nn.Embedding(50, num_pos_feats)#50经验值
        self.col_embed = nn.Embedding(50, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        return pos


def build_position_encoding(cfg):
    N_steps = cfg.MODEL.hidden_dim // 2
    if cfg.MODEL.pos_position_embedding =="v2":
        # TODO find a better way of exposing other arguments
        position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
    elif cfg.MODEL.que_position_embedding =="v3":
        position_embedding = PositionEmbeddingLearned(N_steps)
    else:
        raise ValueError(f"not supported {cfg.position_embedding}")

    return position_embedding
