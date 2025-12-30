"""
MLP模块 - 包含Projector和Classifier
用于对比学习投影和最终分类
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Projector(nn.Module):
    """
    投影头 - 用于对比学习的语义对齐
    将高维特征投影到低维空间，用于计算对比损失
    """
    def __init__(self, input_dim, hidden_dim=None, output_dim=128, dropout=0.1):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = input_dim
        
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        """
        x: [batch, input_dim] - 全局特征
        return: [batch, output_dim] - L2归一化的投影特征
        """
        proj = self.projection(x)
        # L2归一化，使得所有向量在单位超球面上
        proj = F.normalize(proj, p=2, dim=-1)
        return proj


class Classifier(nn.Module):
    """
    分类器 - 用于最终的情感预测
    输入融合后的多模态特征，输出类别logits
    """
    def __init__(self, input_dim, hidden_dims, num_classes, dropout=0.1):
        """
        Args:
            input_dim: 输入特征维度
            hidden_dims: 隐藏层维度列表，例如 [256, 128]
            num_classes: 分类类别数
            dropout: Dropout比率
        """
        super().__init__()
        
        layers = []
        in_dim = input_dim
        
        # 构建隐藏层
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(in_dim, num_classes))
        
        self.classifier = nn.Sequential(*layers)
        
    def forward(self, x):
        """
        x: [batch, input_dim] - 融合后的特征
        return: [batch, num_classes] - 类别logits
        """
        return self.classifier(x)


class SimpleClassifier(nn.Module):
    """
    简单分类器 - 单层MLP
    """
    def __init__(self, input_dim, num_classes, dropout=0.1):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_dim, num_classes)
        )
        
    def forward(self, x):
        return self.classifier(x)


class MLPWithResidual(nn.Module):
    """
    带残差连接的MLP
    """
    def __init__(self, input_dim, hidden_dim, output_dim=None, dropout=0.1):
        super().__init__()
        if output_dim is None:
            output_dim = input_dim
        
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(output_dim)
        
        # 残差连接的投影层（如果维度不匹配）
        self.residual_proj = nn.Linear(input_dim, output_dim) if input_dim != output_dim else None
        
    def forward(self, x):
        identity = x
        
        out = self.linear1(x)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.linear2(out)
        
        # 残差连接
        if self.residual_proj is not None:
            identity = self.residual_proj(identity)
        
        out = self.norm(out + identity)
        return out


# ============= 用于原文方法的专用MLP =============

class DualProjector(nn.Module):
    """
    双投影头 - 为两个模态分别创建投影头
    用于对比学习的语义对齐
    """
    def __init__(self, rcs_dim, jtf_dim, proj_dim=128, dropout=0.1):
        super().__init__()
        self.rcs_projector = Projector(rcs_dim, output_dim=proj_dim, dropout=dropout)
        self.jtf_projector = Projector(jtf_dim, output_dim=proj_dim, dropout=dropout)
        
    def forward(self, rcs_feat, jtf_feat):
        """
        rcs_feat: [batch, rcs_dim]
        jtf_feat: [batch, jtf_dim]
        return: (rcs_proj, jtf_proj) - 两个投影特征
        """
        rcs_proj = self.rcs_projector(rcs_feat)
        jtf_proj = self.jtf_projector(jtf_feat)
        return rcs_proj, jtf_proj


class MultimodalClassifier(nn.Module):
    """
    多模态分类器 - 接收多个模态的特征并融合后分类
    这是论文中最后使用的分类器结构
    """
    def __init__(self, rcs_dim, jtf_dim, bottleneck_dim, 
                 hidden_dims, num_classes, dropout=0.1):
        """
        Args:
            rcs_dim: RCS特征维度
            jtf_dim: JTF特征维度
            bottleneck_dim: 瓶颈特征维度
            hidden_dims: 隐藏层维度列表
            num_classes: 分类类别数
            dropout: Dropout比率
        """
        super().__init__()
        
        # 拼接后的总维度
        input_dim = rcs_dim + jtf_dim + bottleneck_dim
        
        # 使用标准分类器
        self.classifier = Classifier(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            num_classes=num_classes,
            dropout=dropout
        )
        
    def forward(self, rcs_feat, jtf_feat, bottleneck_feat):
        """
        rcs_feat: [batch, rcs_dim]
        jtf_feat: [batch, jtf_dim]
        bottleneck_feat: [batch, bottleneck_dim]
        return: [batch, num_classes]
        """
        # 拼接所有特征
        fused_feat = torch.cat([rcs_feat, jtf_feat, bottleneck_feat], dim=-1)
        # 分类
        logits = self.classifier(fused_feat)
        return logits


# ============= 测试代码 =============

if __name__ == '__main__':
    # 测试投影头
    print("Testing Projector...")
    projector = Projector(input_dim=128, output_dim=64)
    x = torch.randn(8, 128)
    proj = projector(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {proj.shape}")
    print(f"L2 norm: {torch.norm(proj, p=2, dim=-1)}")  # 应该都是1.0
    
    # 测试分类器
    print("\nTesting Classifier...")
    classifier = Classifier(input_dim=384, hidden_dims=[256, 128], num_classes=10)
    x = torch.randn(8, 384)
    logits = classifier(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")
    
    # 测试双投影头
    print("\nTesting DualProjector...")
    dual_proj = DualProjector(rcs_dim=128, jtf_dim=128, proj_dim=64)
    rcs_feat = torch.randn(8, 128)
    jtf_feat = torch.randn(8, 128)
    rcs_proj, jtf_proj = dual_proj(rcs_feat, jtf_feat)
    print(f"RCS proj shape: {rcs_proj.shape}")
    print(f"JTF proj shape: {jtf_proj.shape}")
    
    # 测试多模态分类器
    print("\nTesting MultimodalClassifier...")
    mm_classifier = MultimodalClassifier(
        rcs_dim=128, jtf_dim=128, bottleneck_dim=128,
        hidden_dims=[256, 128], num_classes=10
    )
    rcs_feat = torch.randn(8, 128)
    jtf_feat = torch.randn(8, 128)
    bottleneck_feat = torch.randn(8, 128)
    logits = mm_classifier(rcs_feat, jtf_feat, bottleneck_feat)
    print(f"Output logits shape: {logits.shape}")
    
    print("\n✓ All MLP modules working correctly!")
