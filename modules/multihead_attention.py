import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
import sys
from typing import Optional, Tuple
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_
# Code adapted from the fairseq repo.
from torch import Tensor

class MultiheadAttention(nn.Module):
    """Multi-headed attention.
    See "Attention Is All You Need" for more details.
    """

    def __init__(self, embed_dim, num_heads, attn_dropout=0.,
                 bias=True, add_bias_kv=False, add_zero_attn=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.in_proj_weight = Parameter(torch.Tensor(3 * embed_dim, embed_dim))
        self.register_parameter('in_proj_bias', None)
        if bias:
            self.in_proj_bias = Parameter(torch.Tensor(3 * embed_dim))
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def forward(self, query, key, value, attn_mask=None):

        """Input shape: Time x Batch x Channel
        Self-attention can be implemented by passing in the same arguments for
        query, key and value. Timesteps can be masked by supplying a T x T mask in the
        `attn_mask` argument. Padding elements can be excluded from
        the key by passing a binary ByteTensor (`key_padding_mask`) with shape:
        batch x src_len, where padding elements are indicated by 1s.
        """
        qkv_same = query.data_ptr() == key.data_ptr() == value.data_ptr()
        kv_same = key.data_ptr() == value.data_ptr()

        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        assert key.size() == value.size()

        aved_state = None

        if qkv_same:
            # self-attention
            q, k, v = self.in_proj_qkv(query)
        elif kv_same:
            # encoder-decoder attention
            q = self.in_proj_q(query)

            if key is None:
                assert value is None
                k = v = None
            else:
                k, v = self.in_proj_kv(key)
        else:
            q = self.in_proj_q(query)
            k = self.in_proj_k(key)
            v = self.in_proj_v(value)
        q = q * self.scaling

        if self.bias_k is not None:
            assert self.bias_v is not None
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        src_len = k.size(1)

        if self.add_zero_attn:
            src_len += 1
            k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
            v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if attn_mask is not None:
            try:
                attn_weights += attn_mask.unsqueeze(0)
            except:
                print(attn_weights.shape)
                print(attn_mask.unsqueeze(0).shape)
                assert False

        attn_weights = F.softmax(attn_weights.float(), dim=-1).type_as(attn_weights)
        # attn_weights = F.relu(attn_weights)
        # attn_weights = attn_weights / torch.max(attn_weights)
        attn_weights = F.dropout(attn_weights, p=self.attn_dropout, training=self.training)

        attn = torch.bmm(attn_weights, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]

        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)

        # average attention weights over heads
        attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
        attn_weights = attn_weights.sum(dim=1) / self.num_heads
        return attn, attn_weights

    def in_proj_qkv(self, query):
        return self._in_proj(query).chunk(3, dim=-1)

    def in_proj_kv(self, key):
        return self._in_proj(key, start=self.embed_dim).chunk(2, dim=-1)

    def in_proj_q(self, query, **kwargs):
        return self._in_proj(query, end=self.embed_dim, **kwargs)

    def in_proj_k(self, key):
        return self._in_proj(key, start=self.embed_dim, end=2 * self.embed_dim)

    def in_proj_v(self, value):
        return self._in_proj(value, start=2 * self.embed_dim)

    def _in_proj(self, input, start=0, end=None, **kwargs):
        weight = kwargs.get('weight', self.in_proj_weight)
        bias = kwargs.get('bias', self.in_proj_bias)
        weight = weight[start:end, :]
        if bias is not None:
            bias = bias[start:end]
        return F.linear(input, weight, bias)

# class MyMultiheadAttention(nn.Module):
#     def __init__(self, embed_dim, num_heads, dropout=0., bias=True):
#         super(MyMultiheadAttention, self).__init__()
#         """
#         :param embed_dim:   ???????????????????????????????????????d_model?????????????????????????????????512
#         :param num_heads:   ????????????????????????????????????????????????????????????nhead????????? ?????????????????? 8
#         :param bias:        ???????????????????????????????????????????????????????????????????????????????????????
#         """
#         self.embed_dim = embed_dim  # ?????????d_model??????
#         self.head_dim = embed_dim // num_heads  # head_dim ????????????d_k,d_v
#         self.kdim = self.head_dim
#         self.vdim = self.head_dim
#         self.num_heads = num_heads  # ????????????
#         self.dropout = dropout
#         assert self.head_dim * num_heads == self.embed_dim, "embed_dim ?????? num_heads???????????????"
#         # ???????????????????????????????????????  d_k = d_v = d_model/n_head ??????
#         self.q_proj_weight = Parameter(torch.Tensor(embed_dim, embed_dim))  # embed_dim = kdim * num_heads
#         # ?????????????????????????????????embed_dim???????????????????????????????????????num_heads???W_q???????????????, ?????????num_heads??????
#         self.k_proj_weight = Parameter(torch.Tensor(embed_dim, embed_dim))  # W_k,  embed_dim = kdim * num_heads
#         self.v_proj_weight = Parameter(torch.Tensor(embed_dim, embed_dim))  # W_v,  embed_dim = vdim * num_heads
#         self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
#         # ??????????????????Z???????????????????????????????????????????????? embed_dim = vdim * num_heads
#     def multi_head_attention_forward(
#         self,
#         query,  # [tgt_len,batch_size, embed_dim]
#         key,  # [src_len, batch_size, embed_dim]
#         value,  # [src_len, batch_size, embed_dim]
#         num_heads,
#         dropout_p,
#         out_proj_weight, # [embed_dim = vdim * num_heads, embed_dim]
#         out_proj_bias,
#         training=True,
#         key_padding_mask=None,  # [batch_size,src_len/tgt_len]
#         q_proj_weight=None,  # [embed_dim,kdim * num_heads]
#         k_proj_weight=None,  # [embed_dim, kdim * num_heads]
#         v_proj_weight=None,  # [embed_dim, vdim * num_heads]
#         attn_mask=None,  # [tgt_len,src_len] 
#         ):
#         # ??????????????? ????????????Q???K???V
#         q = F.linear(query, q_proj_weight)
#         #  [tgt_len,batch_size,embed_dim] x [embed_dim,kdim * num_heads] = [tgt_len,batch_size,kdim * num_heads]
#         k = F.linear(key, k_proj_weight)
#         # [src_len, batch_size,embed_dim] x [embed_dim,kdim * num_heads] = [src_len,batch_size,kdim * num_heads]
#         v = F.linear(value, v_proj_weight)
#         # [src_len, batch_size,embed_dim] x [embed_dim,vdim * num_heads] = [src_len,batch_size,vdim * num_heads]
#         # ??????????????? ???????????????attn_mask????????????
#         tgt_len, bsz, embed_dim = query.size()  # [tgt_len,batch_size, embed_dim]
#         src_len = key.size(0)
#         head_dim = embed_dim // num_heads  # num_heads * head_dim = embed_dim
#         scaling = float(head_dim) ** -0.5
#         q = q * scaling  # [query_len,batch_size,kdim * num_heads]

#         if attn_mask is not None:  # [tgt_len,src_len] or [num_heads*batch_size,tgt_len, src_len]
#             if attn_mask.dim() == 2:
#                 attn_mask = attn_mask.unsqueeze(0)  # [1, tgt_len,src_len]
#                 if list(attn_mask.size()) != [1, query.size(0), key.size(0)]:
#                     raise RuntimeError('The size of the 2D attn_mask is not correct.')
#             elif attn_mask.dim() == 3:
#                 if list(attn_mask.size()) != [bsz * num_heads, query.size(0), key.size(0)]:
#                     raise RuntimeError('The size of the 3D attn_mask is not correct.')
#         # ?????? atten_mask ?????????????????????3D
#         #??????????????????????????????5-6????????????????????????7????????????????????????8-15????????????????????????attn_mask?????????????????????????????????????????????????????????Masked Multi-Head Attention????????????

#         # ??????????????? ?????????????????????????????????
#         q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
#         # [batch_size * num_heads,tgt_len,kdim]
#         # ???????????????num_heads?????????????????????????????????????????????????????????????????????????????????????????? ??????????????????0???1????????????
#         k = k.contiguous().view(-1, bsz*num_heads, head_dim).transpose(0,1)#[batch_size * num_heads,src_len,kdim]
#         v = v.contiguous().view(-1, bsz*num_heads, head_dim).transpose(0,1)#[batch_size * num_heads,src_len,vdim]
#         attn_output_weights = torch.bmm(q, k.transpose(1, 2))
#         # [batch_size * num_heads,tgt_len,kdim] x [batch_size * num_heads, kdim, src_len]
#         # =  [batch_size * num_heads, tgt_len, src_len]  ??????num_heads???QK???????????????????????????
#         # ??????????????? ????????????????????????
#         if attn_mask is not None:
#             attn_output_weights += attn_mask  # [batch_size * num_heads, tgt_len, src_len]
#         if key_padding_mask is not None:
#             attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
#             # ?????? [batch_size, num_heads, tgt_len, src_len]?????????
#             attn_output_weights = attn_output_weights.masked_fill(
#                 key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf')) 
#             # ??????????????????[batch_size,src_len]??????[batch_size,1,1,src_len]
#             attn_output_weights = attn_output_weights.view(bsz * num_heads, tgt_len,src_len)  
#             # [batch_size * num_heads, tgt_len, src_len]
#         attn_output_weights = F.softmax(attn_output_weights, dim=-1)# [batch_size * num_heads, tgt_len, src_len]
#         attn_output_weights = F.dropout(attn_output_weights, p=dropout_p, training=training)
#         attn_output = torch.bmm(attn_output_weights, v)
#         # Z = [batch_size * num_heads, tgt_len, src_len]  x  [batch_size * num_heads,src_len,vdim]
#         # = # [batch_size * num_heads,tgt_len,vdim]
#         # ??????num_heads???Attention(Q,K,V)??????

#         attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
#         # ???transpose??? [tgt_len, batch_size* num_heads ,kdim]
#         # ???view??? [tgt_len,batch_size,num_heads*kdim]
#         attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)

#         Z = F.linear(attn_output, out_proj_weight, out_proj_bias)
#         # ??????????????????z  ???????????????Z  [tgt_len,batch_size,embed_dim]
#         return Z, attn_output_weights.sum(dim=1) / num_heads  # ???num_heads????????????????????????????????????????????????
#     def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
#         """
#         ????????????????????????query, key, value ???????????????????????? 
#         ????????? ?????????????????????????????????????????? 
#         ???????????????????????? key,value????????? memory, query?????????tgt
#         :param query: # [tgt_len, batch_size, embed_dim], tgt_len ???????????????????????????
#         :param key:  #  [src_len, batch_size, embed_dim], src_len ????????????????????????
#         :param value: # [src_len, batch_size, embed_dim], src_len ????????????????????????
#         :param attn_mask: # [tgt_len,src_len] or [num_heads*batch_size,tgt_len, src_len]
#                 ????????????????????????????????????????????????????????????????????????????????????????????????mask????????????????????????????????????????????????
#         :param key_padding_mask: [batch_size, src_len], src_len ????????????????????????
#         :return:
#         attn_output: [tgt_len, batch_size, embed_dim]
#         attn_output_weights: # [batch_size, tgt_len, src_len]
#         """
#         return multi_head_attention_forward(query, key, value, self.num_heads,
#                                             self.dropout, self.out_proj.weight, self.out_proj.bias,
#                                             training=self.training,
#                                             key_padding_mask=key_padding_mask,
#                                             q_proj_weight=self.q_proj_weight,
#                                             k_proj_weight=self.k_proj_weight,
#                                             v_proj_weight=self.v_proj_weight,
#                                             attn_mask=attn_mask)