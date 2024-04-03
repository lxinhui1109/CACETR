# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor


class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=4,
                 num_decoder_layers=4, dim_feedforward=2048, dropout=0.3,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask,query_embed, pos_embed,decoder_mask=None):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        # 将降维后的src转换维度[NxCxHxW]->[HWxNxC]
        src = src.flatten(2).permute(2, 0, 1)#[1,256,12,28]->[336,1,256]
        # 将位置编码转换维度[NxCxHxW]->[HWxNxC],
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)#[1,256,12,28]->[336,1,256]
        # 词嵌入向量由[num_embeddings, embedding_dim]->[num_embeddings, N, embedding_dim]
        #query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)# 即[100,256]->[100,2,256]

        #mask和exe都是decoder才会用到
        mask = mask.flatten(1)# 将mask[1，12，28]->[1，336]
        #exe = torch.zeros_like(query_embed)#[100,2,256]


        # memory shape与src相同, 进入到transformerencoderlayer中
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)#mask是padding出来的 它们不需要计算注意力
        # hs 为decoder的输出
        hs = self.decoder(query_embed, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, query_pos=query_embed,exe_mask=decoder_mask)
        # 最后返回的hs交换了第二和第三维

        return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w)


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src
        #for循环 encoder有6层，循环6次
        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)
        # output[650,2,256]
        return output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, exe, memory,
                exe_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                exe_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = exe

        intermediate = []
    #for循环 decoder层    exe每经过一层是会发生变化的，而memory一直都是encoder输出的特征图
        for layer in self.layers:
            output = layer(output, memory, exe_mask=exe_mask,
                           memory_mask=memory_mask,
                           exe_key_padding_mask=exe_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            # 由于return_intermediate设置为True,所以decoder每一层的输出都被保
            # 存在intermediate列表中,decoder每一层的输出为[100,2,256],最终
            # 返回的是将intermediate进行stack的结果，所以最后的输出为[6,100,2,256]
            return torch.stack(intermediate)

        return output.unsqueeze(0)


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.3,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        # q,k由最初输入的src加上pos的位置编码构成,且q=k
        q = k = self.with_pos_embed(src, pos)

        # self_attn是nn.MultiheadAttention 直接用现成的东西
        # 进入自注意力层,src2 = softmax(q*kt/sqrt(dk))*v,其中dk=32
        # 进入自注意力层后会对q,k,v进行reshape,由输入的[HWxNxC]->[NXnum_heads,HxW,head_dim]
        # 自注意力层的输出依旧是[650,2,256]
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,#attn_mask=src_mask意思是 我们不需要考虑 预测当前值要把后面的mask，也就是前面的语句后面的语句我们都可以看
                              key_padding_mask=src_key_padding_mask)[0]#key_padding_mask=src_key_padding_mask意思是，这里面是true值的是我们需要mask掉的，不看它的自注意力
                            #[0]两个返回值：自注意力层的输出，自注意力权重，我们只需要第一个
        #transformer encoder里的操作一样
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.3,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, exe, memory,
                     exe_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     exe_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        # q,k由最初输入的exe加上query_pos的词嵌入向量构成,且q=k,shape为[100,2,256]，一开始只有位置上的信息
        # 其中exe（只有第一层） 因为一开始query向量还不知道怎样利用encoder输出的memory进行学习
        q = k = self.with_pos_embed(exe, query_pos)
        # q=k=exe
        # 自注意力层,exe2 = softmax(q*kt/sqrt(dk))*v,其中dk=32  这部分是自己在做，没有用到encoder的输出
        # 进入自注意力层后会对q,k,v进行reshape,由输入的[HWxNxC]->[NXnum_heads,HxW,head_dim],即[100,2,256]->[16,100,32]
        # 自注意力层的输出依旧是[100,2,256]
        exe2 = self.self_attn(q, k, value=exe, attn_mask=exe_mask,#这100个都有用 没有mask的
                              key_padding_mask=exe_key_padding_mask)[0]
        exe = exe + self.dropout1(exe2)
        exe = self.norm1(exe)

        # 多头注意力层,计算方式相同exe2 = softmax(q*kt/sqrt(dk))*v,其中dk=32
        # 但是出入的shape发生变化,memory是encoder的输出,shape为[650,2,256],用它作为k,v,k还要加上位置编码
        # 多头自注意力层同样对q,k,v进行reshape,由输入的[HWxNxC]->[NXnum_heads,HxW,head_dim]
        # 即q:[100,2,256]->[16,100,32]与k,v:[650,2,256]->[16,650,32]
        # 多头注意力层的输出依旧是[100,2,256]
        exe2 = self.multihead_attn(query=self.with_pos_embed(exe, query_pos), #exe是decoder的q，去查memory的信息
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        exe = exe + self.dropout2(exe2)
        exe = self.norm2(exe)
        exe2 = self.linear2(self.dropout(self.activation(self.linear1(exe))))
        exe = exe + self.dropout3(exe2)
        exe = self.norm3(exe)
        return exe

    def forward_pre(self, exe, memory,
                    exe_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    exe_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        exe2 = self.norm1(exe)
        q = k = self.with_pos_embed(exe2, query_pos)
        # q = k = exe2
        exe2 = self.self_attn(q, k, value=exe2, attn_mask=exe_mask,
                              key_padding_mask=exe_key_padding_mask)[0]
        exe = exe + self.dropout1(exe2)
        exe2 = self.norm2(exe)
        exe2 = self.multihead_attn(query=self.with_pos_embed(exe2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        exe = exe + self.dropout2(exe2)
        exe2 = self.norm3(exe)
        exe2 = self.linear2(self.dropout(self.activation(self.linear1(exe2))))
        exe = exe + self.dropout3(exe2)
        return exe

    def forward(self, exe, memory,
                exe_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                exe_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(exe, memory, exe_mask, memory_mask,
                                    exe_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(exe, memory, exe_mask, memory_mask,
                                 exe_key_padding_mask, memory_key_padding_mask, pos, query_pos)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(cfg):
    return Transformer(
        d_model=cfg.MODEL.hidden_dim,
        dropout=cfg.MODEL.dropout,
        nhead=cfg.MODEL.nheads,
        dim_feedforward=cfg.MODEL.dim_feedforward,
        num_encoder_layers=cfg.MODEL.enc_layers,
        num_decoder_layers=cfg.MODEL.dec_layers,
        normalize_before=cfg.MODEL.pre_norm,
        return_intermediate_dec=True,
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
