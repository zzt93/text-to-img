import torch
import torch.nn as nn
from typing import Optional, Tuple
import torch.nn.functional as F

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
            t, w = super().forward(query, key, value, key_padding_mask, need_weights, attn_mask, is_causal)
            return t, w, None

        # 自注意力模式：query/key/value 相同
        if key is None:
            key = query
        if value is None:
            value = query


        # 提取当前输入的键值对（未拼接缓存）
        q, k_proj, v_proj = self._compute_kv(key, value)

        k, v = k_proj, k_proj
        # 如果提供了缓存，拼接历史缓存
        if cache is not None:
            k = torch.cat([cache[0], k], dim=1)  # 沿序列维度拼接 (batch, seq_len, ...)
            v = torch.cat([cache[1], v], dim=1)


        # 调用原始注意力计算
        bsz, tgt_len, embed_dim = query.shape
        num_heads = self.num_heads
        head_dim = self.head_dim
        src_len = k.size(1)

        q = q.view(bsz, num_heads, tgt_len, head_dim)
        k = k.view(bsz, num_heads, src_len, head_dim)
        v = v.view(bsz, num_heads, src_len, head_dim)

        attn_output = F.scaled_dot_product_attention(
            q, k, v, attn_mask, 0.0, is_causal
        )
        attn_output = (
            attn_output.permute(2, 0, 1, 3).contiguous().view(bsz * tgt_len, embed_dim)
        )

        attn_output = F.linear(attn_output, self.out_proj_weight, self.out_proj_bias)
        attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))

        new_cache = (k_proj.detach(), v_proj.detach())

        return attn_output, None, new_cache

    def _compute_kv(self, key: torch.Tensor, value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """计算键值对（分离投影操作）"""
        # 复用 MultiheadAttention 的投影权重
        if self.in_proj_weight is not None:
            # 合并计算 q/k/v 的投影
            qkv = F.linear(key, self.in_proj_weight, self.in_proj_bias)
            q, k, v = qkv.chunk(3, dim=-1)
        else:
            # 分别计算 k/v 的投影
            k = F.linear(key, self.k_proj_weight, self.k_proj_bias)
            v = F.linear(value, self.v_proj_weight, self.v_proj_bias)
            q = F.linear(key, self.q_proj_weight, self.q_proj_bias)

        return q, k, v