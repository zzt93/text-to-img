import torch
import torch.nn as nn
from typing import Optional, Tuple
import torch.nn.functional as F

import config


class MultiheadAttentionWithCache(nn.MultiheadAttention):
    def __init__(self, embed_dim: int, num_heads: int, **kwargs):
        super().__init__(embed_dim, num_heads, **kwargs)
        # 初始化缓存占位符
        self.k_cache: Optional[torch.Tensor] = None
        self.v_cache: Optional[torch.Tensor] = None
        self.cache_enabled = False  # 是否启用缓存的标志

    def enable_cache(self) -> None:
        """启用缓存"""
        self.cache_enabled = True

    def disable_cache(self) -> None:
        """禁用并清空缓存"""
        self.cache_enabled = False
        self.k_cache = None
        self.v_cache = None

    def forward(
            self,
            query: torch.Tensor,
            key: Optional[torch.Tensor] = None,
            value: Optional[torch.Tensor] = None,
            key_padding_mask: Optional[torch.Tensor] = None,
            need_weights: bool = False,
            attn_mask: Optional[torch.Tensor] = None,
            is_causal: bool = False,  # PyTorch 2.0+ 新增参数
            cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        扩展后的前向传播，支持缓存返回

        Args:
            cache: 输入的缓存 (k_cache, v_cache)

        Returns:
            attn_output: 注意力输出
            attn_weights: 注意力权重（如果 need_weights=True）
            cache: 更新后的缓存 (new_k_cache, new_v_cache)
        """
        # 如果未启用缓存，直接调用原始方法
        if not self.cache_enabled:
            t, w = super().forward(query, query, query, key_padding_mask, need_weights, attn_mask, is_causal)
            return t, w, None

        # if batch_size == 1, remove this dim
        if query.size(0) == 1:
            query = query.squeeze(0)
        # 提取当前输入的键值对（未拼接缓存）
        q, k_proj, v_proj = self._compute_kv(query, cache is not None)

        k, v = k_proj, v_proj
        # 如果提供了缓存，拼接历史缓存
        if cache is not None:
            k = torch.cat([cache[0], k], dim=0)  # 沿序列维度拼接 (seq_len, hid_dim)
            v = torch.cat([cache[1], v], dim=0)

        # qq, kk, vv = self._compute_kv(query, False)
        # assert (kk[-1,:] == k[-1,:]).all()
        # assert (vv[-1,:] == v[-1,:]).all()

        # 调用原始注意力计算
        bsz = 1
        tgt_len = q.size(0)
        if attn_mask.size(0) != tgt_len:
            assert False, "attn_mask.size(0) != tgt_len"
        src_len = k.size(0)
        embed_dim = q.size(-1)
        num_heads = self.num_heads
        head_dim = self.head_dim

        q = q.view(bsz, tgt_len, num_heads, head_dim).transpose(1, 2)
        k = k.view(bsz, src_len, num_heads, head_dim).transpose(1, 2)
        v = v.view(bsz, src_len, num_heads, head_dim).transpose(1, 2)
        attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)

        # if cache is not None:
        # print(q, k, v)
        attn_output = F.scaled_dot_product_attention(
            q, k, v, attn_mask, 0.0, False
        )
        attn_output = (
            attn_output.permute(2, 0, 1, 3).contiguous().view(bsz * tgt_len, embed_dim)
        )

        attn_output = F.linear(attn_output, self.out_proj.weight, self.out_proj.bias)
        attn_output = attn_output.view(bsz, tgt_len, attn_output.size(1))

        new_cache = (k_proj.detach(), v_proj.detach())

        return attn_output, None, new_cache

    def _compute_kv(self, query: torch.Tensor, has_cache: bool) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """计算键值对（分离投影操作）"""
        # 复用 MultiheadAttention 的投影权重
        if self.in_proj_weight is not None:
            # 合并计算 q/k/v 的投影
            if has_cache:
                # 使用 [-1:, :] 来计算最后一行，会有一点数值误差，但是不影响
                qkv = F.linear(query[-1:, :], self.in_proj_weight, self.in_proj_bias)[-1,:]
            else:
                qkv = F.linear(query, self.in_proj_weight, self.in_proj_bias)
            q, k, v = qkv.chunk(3, dim=-1)
            # print(query, self.in_proj_weight, self.in_proj_bias, q, k, v)
            if q.dim() == 1:
                q = q.unsqueeze(0)
                k = k.unsqueeze(0)
                v = v.unsqueeze(0)

            # a = F.linear(query[-1, :], self.in_proj_weight, self.in_proj_bias)
            # aq, ak, av = a.chunk(3, dim=-1)
            # print(torch.all(aq == q[-1, :]), torch.all(ak == k[-1, :]), torch.all(av == v[-1, :]))
            #
            # b = F.linear(query[-2:, :], self.in_proj_weight, self.in_proj_bias)[-1, :]
            # bq, bk, bv = b.chunk(3, dim=-1)
            # print(torch.all(bq == q[-1, :]), torch.all(bk == k[-1, :]), torch.all(bv == v[-1, :]))
            #
            # c = F.linear(query[-3:, :], self.in_proj_weight, self.in_proj_bias)[-1, :]
            # cq, ck, cv = c.chunk(3, dim=-1)
            # print(torch.all(cq == q[-1, :]), torch.all(ck == k[-1, :]), torch.all(cv == v[-1, :]))
            #
            # d = F.linear(query, self.in_proj_weight, self.in_proj_bias)[-1, :]
            # dq, dk, dv = d.chunk(3, dim=-1)
            # print(torch.all(dq == q[-1, :]), torch.all(dk == k[-1, :]), torch.all(dv == v[-1, :]))

            # e = F.linear(query[-10:, :], self.in_proj_weight, self.in_proj_bias)[-1, :]
            # eq, ek, ev = e.chunk(3, dim=-1)
            # print(torch.all(eq == q[-1, :]), torch.all(ek == k[-1, :]), torch.all(ev == v[-1, :]))
            #
            # f = F.linear(query.squeeze(0), self.in_proj_weight, self.in_proj_bias)[-1, :]
            # fq, fk, fv = f.chunk(3, dim=-1)
            # print(torch.all(fq == q[-1, :]), torch.all(fk == k[-1, :]), torch.all(fv == v[-1, :]))
            #
            # g = F.linear(query.squeeze(0)[-1,:], self.in_proj_weight, self.in_proj_bias)
            # gq, gk, gv = g.chunk(3, dim=-1)
            # print(torch.all(gq == q[-1, :]), torch.all(gk == k[-1, :]), torch.all(gv == v[-1, :]))
            #
            # h = query @ self.in_proj_weight.transpose(0,1) + self.in_proj_bias
            # hq, hk, hv = h.chunk(3, dim=-1)
            # print(torch.all(hq == q[-1, :]), torch.all(hk == k[-1, :]), torch.all(hv == v[-1, :]))

            # if key.size(1) == 1:
            #     print(key, self.in_proj_weight, self.in_proj_bias, k)
        else:
            # 分别计算 k/v 的投影
            k = F.linear(query, self.k_proj_weight, self.k_proj_bias)
            q = F.linear(query, self.q_proj_weight, self.q_proj_bias)
            v = F.linear(query, self.v_proj_weight, self.v_proj_bias)

        return q, k, v