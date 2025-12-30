"""
RCS和JTF编码器
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.layers import TransformerEncoderLayer


class RCSEncoder(nn.Module):
    """
    RCS序列编码器
    输入: [batch, 1, 256] - RCS时域序列
    输出: [batch, seq_len, hidden_dim] - 编码后的序列特征
    """
    def __init__(self, rcs_dim=256, hidden_dim=128, num_layers=2, num_heads=4, dropout=0.1):
        super().__init__()
        self.rcs_dim = rcs_dim
        self.hidden_dim = hidden_dim
        
        # 1D卷积用于局部特征提取
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        
        # 位置编码
        self.pos_embedding = nn.Parameter(torch.randn(1, rcs_dim, hidden_dim))
        
        # Transformer编码器
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayer(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        x: [batch, 1, 256]
        """
        batch_size = x.size(0)
        
        # 1D卷积特征提取
        x = self.conv_layers(x)  # [batch, hidden_dim, 256]
        x = x.transpose(1, 2)     # [batch, 256, hidden_dim]
        
        # 添加位置编码
        x = x + self.pos_embedding
        x = self.dropout(x)
        
        # Transformer编码
        for layer in self.transformer_layers:
            x = layer(x)
        
        return x  # [batch, 256, hidden_dim]


class JTFEncoder(nn.Module):
    """
    JTF时频图编码器
    输入: [batch, 1, 256, 256] - JTF时频图
    输出: [batch, seq_len, hidden_dim] - 编码后的序列特征
    """
    def __init__(self, jtf_size=256, hidden_dim=128, num_layers=2, num_heads=4, dropout=0.1):
        super().__init__()
        self.jtf_size = jtf_size
        self.hidden_dim = hidden_dim
        
        # 2D卷积用于图像特征提取
        self.conv_layers = nn.Sequential(
            # Stage 1
            nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 256 -> 128 -> 64
            
            # Stage 2
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 64 -> 32 -> 16
            
            # Stage 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            # Stage 4
            nn.Conv2d(128, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU()
        )
        
        # 自适应池化到固定大小
        self.adaptive_pool = nn.AdaptiveAvgPool2d((16, 16))  # 输出 16x16
        
        # 位置编码 (16*16=256个位置)
        self.pos_embedding = nn.Parameter(torch.randn(1, 256, hidden_dim))
        
        # Transformer编码器
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayer(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        x: [batch, 1, 256, 256]
        """
        batch_size = x.size(0)
        
        # 2D卷积特征提取
        x = self.conv_layers(x)           # [batch, hidden_dim, H, W]
        x = self.adaptive_pool(x)         # [batch, hidden_dim, 16, 16]
        
        # 展平为序列
        x = x.flatten(2)                  # [batch, hidden_dim, 256]
        x = x.transpose(1, 2)             # [batch, 256, hidden_dim]
        
        # 添加位置编码
        x = x + self.pos_embedding
        x = self.dropout(x)
        
        # Transformer编码
        for layer in self.transformer_layers:
            x = layer(x)
        
        return x  # [batch, 256, hidden_dim]


# ProjectionHead 已移至 MLP.py
# 请使用: from MLP import Projector
