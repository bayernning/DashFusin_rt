"""
优化版RCS和TF编码器 - 针对小数据集 + 信号数据
特点:
1. 轻量化设计（避免过拟合）
2. 频域特征增强
3. 改进的位置编码
4. 残差连接 + 注意力机制
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================================
#                           位置编码模块
# ============================================================================

class RotaryPositionalEmbedding(nn.Module):
    """
    旋转位置编码 (RoPE) - 相对位置感知
    优势: 
    - 泛化到不同序列长度
    - 捕获相对位置关系
    - 无额外参数
    """
    def __init__(self, dim, max_seq_len=512, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
    def forward(self, x):
        """
        x: [B, L, D]
        return: [B, L, D] 带旋转位置编码的特征
        """
        seq_len = x.shape[1]
        device = x.device
        
        # 生成位置编码
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)  # [L, D/2]
        emb = torch.cat([freqs, freqs], dim=-1)  # [L, D]
        
        cos_emb = emb.cos()[None, :, :]  # [1, L, D]
        sin_emb = emb.sin()[None, :, :]
        
        # 旋转变换
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        
        x_rotated_even = x_even * cos_emb[..., 0::2] - x_odd * sin_emb[..., 1::2]
        x_rotated_odd = x_even * sin_emb[..., 0::2] + x_odd * cos_emb[..., 1::2]
        
        # 交织回去
        x_rotated = torch.stack([x_rotated_even, x_rotated_odd], dim=-1)
        x_rotated = x_rotated.flatten(-2)
        
        return x_rotated


class PositionalEncoding2D(nn.Module):
    """
    2D位置编码 - 用于时频图
    保留时间轴和频率轴的位置信息
    """
    def __init__(self, channels, height, width):
        super().__init__()
        self.pos_encoding = nn.Parameter(torch.randn(1, channels, height, width) * 0.02)
        
    def forward(self, x):
        """x: [B, C, H, W]"""
        return x + self.pos_encoding


# ============================================================================
#                           注意力机制模块
# ============================================================================

class ChannelAttention(nn.Module):
    """
    通道注意力 (SENet风格)
    动态调整不同通道的重要性
    """
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        """x: [B, C, L]"""
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class ChannelAttention2D(nn.Module):
    """2D通道注意力"""
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        """x: [B, C, H, W]"""
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


# ============================================================================
#                           残差块模块
# ============================================================================

class ResidualBlock1D(nn.Module):
    """
    1D残差块 + 通道注意力
    用于RCS编码器
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dropout=0.1):
        super().__init__()
        padding = kernel_size // 2
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 1, padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout2 = nn.Dropout(dropout)
        
        # 通道注意力
        self.ca = ChannelAttention(out_channels)
        
        # 残差连接
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride),
                nn.BatchNorm1d(out_channels)
            )
        
    def forward(self, x):
        identity = self.shortcut(x)
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout1(out)
        
        out = self.bn2(self.conv2(out))
        out = self.dropout2(out)
        
        # 通道注意力
        out = self.ca(out)
        
        out += identity
        out = F.relu(out)
        return out


class ResidualBlock2D(nn.Module):
    """
    2D残差块 + 通道注意力
    用于TF编码器
    """
    def __init__(self, in_channels, out_channels, stride=1, dropout=0.1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.dropout1 = nn.Dropout2d(dropout)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout2 = nn.Dropout2d(dropout)
        
        # 通道注意力
        self.ca = ChannelAttention2D(out_channels)
        
        # 残差连接
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = self.shortcut(x)
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout1(out)
        
        out = self.bn2(self.conv2(out))
        out = self.dropout2(out)
        
        # 通道注意力
        out = self.ca(out)
        
        out += identity
        out = F.relu(out)
        return out


# ============================================================================
#                           改进的Transformer层
# ============================================================================

class ImprovedTransformerEncoderLayer(nn.Module):
    """
    改进的Transformer编码器层
    - 使用预归一化 (Pre-LN)
    - 更强的正则化
    """
    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        
        # 多头自注意力
        self.self_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),  # 使用GELU替代ReLU
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        
        # Layer Normalization (Pre-LN)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Pre-LN + Self-Attention
        x_norm = self.norm1(x)
        attn_out, _ = self.self_attn(x_norm, x_norm, x_norm)
        x = x + self.dropout(attn_out)
        
        # Pre-LN + FFN
        x_norm = self.norm2(x)
        ffn_out = self.ffn(x_norm)
        x = x + ffn_out
        
        return x


# ============================================================================
#                           优化的RCS编码器
# ============================================================================

class OptimizedRCSEncoder(nn.Module):
    """
    优化的RCS编码器
    特点:
    1. 时域 + 频域双分支
    2. 轻量多尺度特征
    3. 旋转位置编码
    4. 残差连接 + 通道注意力
    """
    def __init__(self, rcs_dim=256, hidden_dim=128, num_layers=2, num_heads=4, dropout=0.1):
        super().__init__()
        self.rcs_dim = rcs_dim
        self.hidden_dim = hidden_dim
        
        # =====================================================================
        # 分支1: 时域特征提取
        # =====================================================================
        self.time_branch = nn.Sequential(
            # Stage 1: 细粒度特征
            ResidualBlock1D(1, 32, kernel_size=7, dropout=dropout),
            # Stage 2: 中等粒度
            ResidualBlock1D(32, 64, kernel_size=5, dropout=dropout),
            # Stage 3: 粗粒度
            ResidualBlock1D(64, hidden_dim // 2, kernel_size=3, dropout=dropout)
        )
        
        # =====================================================================
        # 分支2: 频域特征提取
        # =====================================================================
        self.freq_branch = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=7, padding=3),  # 2通道: real + imag
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Conv1d(64, hidden_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU()
        )
        
        # =====================================================================
        # 特征融合
        # =====================================================================
        self.fusion_conv = nn.Conv1d(hidden_dim, hidden_dim, 1)
        self.fusion_bn = nn.BatchNorm1d(hidden_dim)
        
        # =====================================================================
        # 旋转位置编码
        # =====================================================================
        self.pos_encoding = RotaryPositionalEmbedding(hidden_dim, rcs_dim)
        
        # =====================================================================
        # Transformer编码器
        # =====================================================================
        self.transformer_layers = nn.ModuleList([
            ImprovedTransformerEncoderLayer(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        x: [batch, 1, 256] - RCS时域信号
        return: [batch, 256, hidden_dim]
        """
        batch_size = x.size(0)
        
        # =====================================================================
        # 1. 时域分支
        # =====================================================================
        time_feat = self.time_branch(x)  # [B, D/2, 256]
        
        # =====================================================================
        # 2. 频域分支
        # =====================================================================
        # FFT变换
        x_freq = torch.fft.rfft(x, dim=-1)  # [B, 1, 129] (复数)
        
        # 分离实部和虚部
        x_freq_real = x_freq.real  # [B, 1, 129]
        x_freq_imag = x_freq.imag  # [B, 1, 129]
        
        # 拼接实部和虚部
        x_freq_combined = torch.cat([x_freq_real, x_freq_imag], dim=1)  # [B, 2, 129]
        
        # 插值到原始长度
        x_freq_combined = F.interpolate(
            x_freq_combined, 
            size=self.rcs_dim, 
            mode='linear', 
            align_corners=False
        )  # [B, 2, 256]
        
        # 频域特征提取
        freq_feat = self.freq_branch(x_freq_combined)  # [B, D/2, 256]
        
        # =====================================================================
        # 3. 时频特征融合
        # =====================================================================
        combined_feat = torch.cat([time_feat, freq_feat], dim=1)  # [B, D, 256]
        fused_feat = self.fusion_conv(combined_feat)
        fused_feat = self.fusion_bn(fused_feat)
        fused_feat = F.relu(fused_feat)
        
        # 转换为序列格式
        x = fused_feat.transpose(1, 2)  # [B, 256, D]
        
        # =====================================================================
        # 4. 旋转位置编码
        # =====================================================================
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # =====================================================================
        # 5. Transformer编码
        # =====================================================================
        for layer in self.transformer_layers:
            x = layer(x)
        
        return x  # [B, 256, D]


# ============================================================================
#                           优化的TF编码器
# ============================================================================

class OptimizedTFEncoder(nn.Module):
    """
    优化的TF编码器
    特点:
    1. 残差卷积 + 通道注意力
    2. 保留空间信息 (16x16)
    3. 2D位置编码
    4. 渐进式降维
    """
    def __init__(self, target_seq_len=256, hidden_dim=128, num_layers=2, num_heads=4, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.target_seq_len = target_seq_len
        
        # =====================================================================
        # 渐进式特征提取 (保留更多空间信息)
        # =====================================================================
        
        # Stage 1: 256 -> 128
        self.stage1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(dropout)
        )
        
        # Stage 2: 128 -> 64
        self.stage2 = nn.Sequential(
            ResidualBlock2D(32, 64, stride=2, dropout=dropout),
        )
        
        # Stage 3: 64 -> 32
        self.stage3 = nn.Sequential(
            ResidualBlock2D(64, 96, stride=2, dropout=dropout),
        )
        
        # Stage 4: 32 -> 16 (最终保留16x16的空间结构)
        self.stage4 = nn.Sequential(
            ResidualBlock2D(96, hidden_dim, stride=2, dropout=dropout),
        )
        
        # =====================================================================
        # 2D位置编码
        # =====================================================================
        self.pos_encoding_2d = PositionalEncoding2D(hidden_dim, 16, 16)
        
        # =====================================================================
        # 空间到序列转换
        # =====================================================================
        # 保留16x16=256的空间位置信息
        self.spatial_proj = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU()
        )
        
        # =====================================================================
        # Transformer编码器
        # =====================================================================
        self.transformer_layers = nn.ModuleList([
            ImprovedTransformerEncoderLayer(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        x: [batch, 1, 256, 256] - TF时频图
        return: [batch, 256, hidden_dim]
        """
        batch_size = x.size(0)
        
        # =====================================================================
        # 1. 渐进式特征提取
        # =====================================================================
        x = self.stage1(x)  # [B, 32, 128, 128]
        x = self.stage2(x)  # [B, 64, 64, 64]
        x = self.stage3(x)  # [B, 96, 32, 32]
        x = self.stage4(x)  # [B, D, 16, 16]
        
        # =====================================================================
        # 2. 添加2D位置编码
        # =====================================================================
        x = self.pos_encoding_2d(x)  # [B, D, 16, 16]
        
        # =====================================================================
        # 3. 空间投影
        # =====================================================================
        x = self.spatial_proj(x)  # [B, D, 16, 16]
        
        # =====================================================================
        # 4. Flatten到序列
        # =====================================================================
        # [B, D, 16, 16] -> [B, D, 256] -> [B, 256, D]
        B, D, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, 256, D]
        
        x = self.dropout(x)
        
        # =====================================================================
        # 5. Transformer编码
        # =====================================================================
        for layer in self.transformer_layers:
            x = layer(x)
        
        return x  # [B, 256, D]


# ============================================================================
#                           测试代码
# ============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("测试优化的编码器")
    print("=" * 70)
    
    # 设置参数
    batch_size = 8
    rcs_dim = 256
    tf_size = 256
    hidden_dim = 128
    
    # 测试RCS编码器
    print("\n【测试RCS编码器】")
    rcs_encoder = OptimizedRCSEncoder(
        rcs_dim=rcs_dim,
        hidden_dim=hidden_dim,
        num_layers=2,
        num_heads=4,
        dropout=0.1
    )
    
    dummy_rcs = torch.randn(batch_size, 1, rcs_dim)
    print(f"输入RCS shape: {dummy_rcs.shape}")
    
    rcs_feat = rcs_encoder(dummy_rcs)
    print(f"输出RCS特征 shape: {rcs_feat.shape}")
    print(f"RCS编码器参数量: {sum(p.numel() for p in rcs_encoder.parameters()):,}")
    
    # 测试TF编码器
    print("\n【测试TF编码器】")
    tf_encoder = OptimizedTFEncoder(
        target_seq_len=256,
        hidden_dim=hidden_dim,
        num_layers=2,
        num_heads=4,
        dropout=0.1
    )
    
    dummy_tf = torch.randn(batch_size, 1, tf_size, tf_size)
    print(f"输入TF shape: {dummy_tf.shape}")
    
    tf_feat = tf_encoder(dummy_tf)
    print(f"输出TF特征 shape: {tf_feat.shape}")
    print(f"TF编码器参数量: {sum(p.numel() for p in tf_encoder.parameters()):,}")
    
    # 总参数量
    total_params = sum(p.numel() for p in rcs_encoder.parameters()) + \
                   sum(p.numel() for p in tf_encoder.parameters())
    print(f"\n【总编码器参数量】: {total_params:,}")
    
    print("\n" + "=" * 70)
    print("✓ 所有测试通过！")
    print("=" * 70)