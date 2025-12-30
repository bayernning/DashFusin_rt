"""
核心层定义 - Attention, Cross-Attention, HBF等
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    """多头自注意力机制"""
    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        assert hidden_dim % num_heads == 0
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.q_linear = nn.Linear(hidden_dim, hidden_dim)
        self.k_linear = nn.Linear(hidden_dim, hidden_dim)
        self.v_linear = nn.Linear(hidden_dim, hidden_dim)
        self.out_linear = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear projections
        Q = self.q_linear(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_linear(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_linear(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_dim)
        
        output = self.out_linear(context)
        return output, attn_weights


class CrossModalAttention(nn.Module):
    """跨模态注意力机制"""
    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(hidden_dim, num_heads, dropout)
        
    def forward(self, query, key, value, mask=None):
        """
        query: 目标模态 [batch, seq_len_q, hidden_dim]
        key, value: 源模态 [batch, seq_len_kv, hidden_dim]
        """
        output, attn_weights = self.attention(query, key, value, mask)
        return output


class MultiCrossAttention(nn.Module):
    """多模态交叉注意力 - 从多个源模态收集信息"""
    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        self.rcs_cross_attn = CrossModalAttention(hidden_dim, num_heads, dropout)
        self.jtf_cross_attn = CrossModalAttention(hidden_dim, num_heads, dropout)
        
    def forward(self, query, rcs_feat, jtf_feat):
        """
        从RCS和JTF特征收集信息到query
        """
        attn_from_rcs = self.rcs_cross_attn(query, rcs_feat, rcs_feat)
        attn_from_jtf = self.jtf_cross_attn(query, jtf_feat, jtf_feat)
        
        # 残差连接
        output = query + attn_from_rcs + attn_from_jtf
        return output


class FeedForward(nn.Module):
    """前馈神经网络"""
    def __init__(self, hidden_dim, ff_dim=None, dropout=0.1):
        super().__init__()
        if ff_dim is None:
            ff_dim = hidden_dim * 4
            
        self.linear1 = nn.Linear(hidden_dim, ff_dim)
        self.linear2 = nn.Linear(ff_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class TransformerEncoderLayer(nn.Module):
    """Transformer编码器层"""
    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(hidden_dim, num_heads, dropout)
        self.ffn = FeedForward(hidden_dim, hidden_dim * 4, dropout)
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-attention
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        
        return x


class HierarchicalBottleneckFusionLayer(nn.Module):
    """层次瓶颈融合层"""
    def __init__(self, hidden_dim, num_bottleneck, num_heads, dropout=0.1):
        super().__init__()
        self.num_bottleneck = num_bottleneck
        self.hidden_dim = hidden_dim
        
        # 阶段1: 从各模态收集信息到瓶颈
        self.multi_cross_attn = MultiCrossAttention(hidden_dim, num_heads, dropout)
        
        # 阶段2: 用瓶颈更新各模态特征
        self.rcs_cross_attn = CrossModalAttention(hidden_dim, num_heads, dropout)
        self.jtf_cross_attn = CrossModalAttention(hidden_dim, num_heads, dropout)
        
        # Feed-forward和LayerNorm
        self.ffn_bottleneck = FeedForward(hidden_dim, dropout=dropout)
        self.ffn_rcs = FeedForward(hidden_dim, dropout=dropout)
        self.ffn_jtf = FeedForward(hidden_dim, dropout=dropout)
        
        self.norm_bottleneck1 = nn.LayerNorm(hidden_dim)
        self.norm_bottleneck2 = nn.LayerNorm(hidden_dim)
        self.norm_rcs1 = nn.LayerNorm(hidden_dim)
        self.norm_rcs2 = nn.LayerNorm(hidden_dim)
        self.norm_jtf1 = nn.LayerNorm(hidden_dim)
        self.norm_jtf2 = nn.LayerNorm(hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, bottleneck, rcs_feat, jtf_feat):
        """
        bottleneck: [batch, num_bottleneck, hidden_dim]
        rcs_feat: [batch, seq_len_rcs, hidden_dim]
        jtf_feat: [batch, seq_len_jtf, hidden_dim]
        """
        # 取前num_bottleneck个token
        bottleneck = bottleneck[:, :self.num_bottleneck, :]
        
        # 阶段1: 从各模态收集信息到瓶颈
        attn_output = self.multi_cross_attn(bottleneck, rcs_feat, jtf_feat)
        bottleneck = self.norm_bottleneck1(bottleneck + self.dropout(attn_output))
        
        # Feed-forward
        ffn_output = self.ffn_bottleneck(bottleneck)
        bottleneck = self.norm_bottleneck2(bottleneck + self.dropout(ffn_output))
        
        # 阶段2: 用瓶颈更新各模态特征
        # 更新RCS特征
        rcs_update = self.rcs_cross_attn(rcs_feat, bottleneck, bottleneck)
        rcs_feat = self.norm_rcs1(rcs_feat + self.dropout(rcs_update))
        rcs_feat = self.norm_rcs2(rcs_feat + self.dropout(self.ffn_rcs(rcs_feat)))
        
        # 更新JTF特征
        jtf_update = self.jtf_cross_attn(jtf_feat, bottleneck, bottleneck)
        jtf_feat = self.norm_jtf1(jtf_feat + self.dropout(jtf_update))
        jtf_feat = self.norm_jtf2(jtf_feat + self.dropout(self.ffn_jtf(jtf_feat)))
        
        return bottleneck, rcs_feat, jtf_feat


class HierarchicalBottleneckFusion(nn.Module):
    """层次瓶颈融合模块"""
    def __init__(self, hidden_dim, num_bottleneck, num_layers, num_heads, dropout=0.1):
        super().__init__()
        self.num_layers = num_layers
        self.num_bottleneck = num_bottleneck
        
        # 初始化瓶颈tokens
        self.bottleneck_tokens = nn.Parameter(
            torch.randn(1, num_bottleneck, hidden_dim)
        )
        
        # 初始Transformer层用于处理multimodal feature
        self.init_transformer = TransformerEncoderLayer(hidden_dim, num_heads, dropout)
        
        # 多层HBF
        self.fusion_layers = nn.ModuleList([
            HierarchicalBottleneckFusionLayer(
                hidden_dim, 
                num_bottleneck // (2 ** i),  # 每层瓶颈数量减半
                num_heads, 
                dropout
            )
            for i in range(num_layers)
        ])
        
    def forward(self, rcs_feat, jtf_feat, aligned_feat):
        """
        rcs_feat: [batch, seq_len_rcs, hidden_dim]
        jtf_feat: [batch, seq_len_jtf, hidden_dim]
        aligned_feat: [batch, seq_len, hidden_dim] - 时间对齐后的多模态特征
        """
        batch_size = rcs_feat.size(0)
        
        # 初始化瓶颈: 从aligned_feat提取信息
        bottleneck = self.init_transformer(aligned_feat)
        bottleneck = bottleneck[:, :self.num_bottleneck, :]  # 取前num_bottleneck个
        
        # 逐层融合
        for layer in self.fusion_layers:
            bottleneck, rcs_feat, jtf_feat = layer(bottleneck, rcs_feat, jtf_feat)
        
        return bottleneck, rcs_feat, jtf_feat