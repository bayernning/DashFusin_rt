"""
DashFusion - 保留Hard Mining + 移除Aligned投影
改动:
1. 保留Hard Mining机制
2. 移除Aligned投影头
3. 简化对比学习：只RCS ↔ TF
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.encoders_v2 import OptimizedRCSEncoder, OptimizedTFEncoder
from model.encoders import RCSEncoder, TFEncoder
from model.layers import CrossModalAttention, HierarchicalBottleneckFusion
from model.MLP import MultimodalClassifier


class FocalLoss(nn.Module):
    """Focal Loss: 专注于难分样本，解决类别不平衡"""
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class SimplifiedDualStreamAlignment(nn.Module):
    """
    简化的双流对齐模块
    改动:
    1. 保留时间对齐 (CrossModalAttention)
    2. 保留RCS和TF的投影头
    3. 移除Aligned投影头 ❌
    """
    def __init__(self, hidden_dim, num_heads, proj_dim=128, dropout=0.1):
        super().__init__()
        
        # =====================================================================
        # 时间对齐: 跨模态注意力
        # =====================================================================
        self.tf_to_rcs = CrossModalAttention(hidden_dim, num_heads, dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        
        # =====================================================================
        # 语义对齐: 只保留RCS和TF的投影头
        # =====================================================================
        self.rcs_projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, proj_dim)
        )
        
        self.tf_projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, proj_dim)
        )
        
        # ❌ 移除 aligned_projector
        
    def temporal_alignment(self, rcs_feat, tf_feat):
        """
        时间对齐: 以RCS为锚点，将TF对齐到RCS
        rcs_feat: [B, 256, D]
        tf_feat: [B, 256, D]
        return: aligned_feat [B, 256, D]
        """
        # TF -> RCS
        tf_to_rcs = self.tf_to_rcs(rcs_feat, tf_feat, tf_feat)
        
        # 融合: RCS + aligned_TF
        aligned_feat = self.norm(rcs_feat + tf_to_rcs)
        
        return aligned_feat
    
    def semantic_alignment(self, rcs_feat, tf_feat):
        """
        语义对齐: 只投影RCS和TF
        rcs_feat: [B, 256, D]
        tf_feat: [B, 256, D]
        return: (rcs_proj, tf_proj)
        """
        # 全局池化
        rcs_global = rcs_feat.mean(dim=1)  # [B, D]
        tf_global = tf_feat.mean(dim=1)    # [B, D]
        
        # 投影并归一化
        rcs_proj = self.rcs_projector(rcs_global)
        rcs_proj = F.normalize(rcs_proj, p=2, dim=1)
        
        tf_proj = self.tf_projector(tf_global)
        tf_proj = F.normalize(tf_proj, p=2, dim=1)
        
        return rcs_proj, tf_proj
    
    def forward(self, rcs_feat, tf_feat):
        """
        前向传播
        return: aligned_feat, rcs_proj, tf_proj
        """
        # 1. 时间对齐
        aligned_feat = self.temporal_alignment(rcs_feat, tf_feat)
        
        # 2. 语义对齐（只投影RCS和TF）
        rcs_proj, tf_proj = self.semantic_alignment(rcs_feat, tf_feat)
        
        # ❌ 不返回aligned_proj
        return aligned_feat, rcs_proj, tf_proj


class SimplifiedContrastiveLoss(nn.Module):
    """
    简化的监督对比损失
    只计算 RCS ↔ TF 的对比
    支持Hard Mining（使用Sample2）
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, features1, features2, labels):
        """
        features1: RCS投影特征 [B, proj_dim]
        features2: TF投影特征 [B, proj_dim]
        labels: 标签 [B]
        
        注意: B可以是32（Anchor）或224（Anchor+Sample2）
        """
        device = features1.device
        batch_size = features1.shape[0]
        
        # 拼接两个模态的特征
        features = torch.cat([features1, features2], dim=0)  # [2B, proj_dim]
        labels = labels.repeat(2)  # [2B]
        
        # 确保特征已归一化
        features = F.normalize(features, p=2, dim=1)
        
        # 计算相似度矩阵
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # 创建mask: 同类为正样本
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        
        # 去除对角线 (自己和自己)
        logits_mask = torch.ones_like(mask).fill_diagonal_(0)
        mask = mask * logits_mask
        
        # 数值稳定性: log-sum-exp trick
        logits_max, _ = similarity_matrix.max(dim=1, keepdim=True)
        logits = similarity_matrix - logits_max.detach()
        
        # 计算exp
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-9)
        
        # 计算平均log-likelihood
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-9)
        
        # 损失
        loss = -mean_log_prob_pos.mean()
        
        return loss


class DashFusion(nn.Module):
    """
    改进版DashFusion
    改动:
    1. 保留Hard Mining (使用Sample2)
    2. 移除Aligned投影头
    3. 简化对比学习 (只RCS ↔ TF)
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 1. 模态编码
        self.rcs_encoder = OptimizedRCSEncoder(
            rcs_dim=config.rcs_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_encoder_layers,
            num_heads=config.num_heads,
            dropout=config.dropout
        )
        
        self.tf_encoder = OptimizedTFEncoder(
            target_seq_len=config.rcs_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_encoder_layers,
            num_heads=config.num_heads,
            dropout=config.dropout
        )
        
        # 2. 简化的双流对齐
        self.dual_alignment = SimplifiedDualStreamAlignment(
            hidden_dim=config.hidden_dim,
            num_heads=config.num_heads,
            proj_dim=128,  # 投影维度
            dropout=config.dropout
        )
        
        # 3. 简化的对比学习损失
        self.contrast_loss = SimplifiedContrastiveLoss(
            temperature=config.temperature
        )
        
        # 4. 分类损失
        class_weights = torch.tensor([1.0, 2.5, 1.0])
        self.cls_loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        # 或使用Focal Loss
        # self.cls_loss_fn = FocalLoss(alpha=1, gamma=2.0)
        
        # 5. 层次瓶颈融合
        self.hierarchical_fusion = HierarchicalBottleneckFusion(
            hidden_dim=config.hidden_dim,
            num_bottleneck=config.num_bottleneck,
            num_layers=config.num_fusion_layers,
            num_heads=config.num_heads,
            dropout=config.dropout
        )
        
        # 6. 分类器
        self.classifier = MultimodalClassifier(
            rcs_dim=config.hidden_dim,
            jtf_dim=config.hidden_dim,
            bottleneck_dim=config.hidden_dim,
            hidden_dims=[config.hidden_dim * 2, config.hidden_dim],
            num_classes=config.num_classes,
            dropout=config.dropout
        )
        
    def forward(self, rcs, tf, labels=None, sample2=None):
        """
        前向传播
        rcs: [B, 1, 256]
        tf: [B, 1, 256, 256]
        labels: [B]
        sample2: dict (可选) - Hard Mining采样的额外样本
        """
        # =====================================================================
        # 1. Anchor模态编码
        # =====================================================================
        rcs_feat = self.rcs_encoder(rcs)      # [B, 256, D]
        tf_feat = self.tf_encoder(tf)         # [B, 256, D]
        
        # =====================================================================
        # 2. Anchor双流对齐
        # =====================================================================
        aligned_feat, rcs_proj, tf_proj = self.dual_alignment(rcs_feat, tf_feat)
        
        # =====================================================================
        # 3. Anchor层次瓶颈融合
        # =====================================================================
        bottleneck, rcs_fused, tf_fused = self.hierarchical_fusion(
            rcs_feat, tf_feat, aligned_feat
        )
        
        # =====================================================================
        # 4. Anchor全局特征提取
        # =====================================================================
        rcs_global = rcs_fused.mean(dim=1)
        tf_global = tf_fused.mean(dim=1)
        bottleneck_global = bottleneck.mean(dim=1)
        
        # =====================================================================
        # 5. Anchor分类
        # =====================================================================
        logits = self.classifier(rcs_global, tf_global, bottleneck_global)
        
        # =====================================================================
        # 6. 计算损失
        # =====================================================================
        if labels is not None:
            # 确保权重在正确的设备上
            if self.cls_loss_fn.weight.device != logits.device:
                self.cls_loss_fn.weight = self.cls_loss_fn.weight.to(logits.device)
            
            # 分类损失（只用Anchor）
            cls_loss = self.cls_loss_fn(logits, labels)
            
            # 对比学习损失 (RCS ↔ TF)
            contrast_loss = 0
            
            # 模式A: 如果提供了Sample2 (Hard Mining模式)
            if sample2 is not None:
                # 提取Sample2特征
                rcs2 = sample2['rcs'].to(rcs.device)
                tf2 = sample2['tf'].to(rcs.device)
                
                # 检查维度
                if rcs2.dim() == 2:
                    rcs2 = rcs2.unsqueeze(1)  # [B*6, 256] -> [B*6, 1, 256]
                
                # 编码Sample2
                rcs2_feat = self.rcs_encoder(rcs2)
                tf2_feat = self.tf_encoder(tf2)
                
                # 对齐Sample2（只需要投影，不需要时间对齐）
                _, rcs2_proj, tf2_proj = self.dual_alignment(rcs2_feat, tf2_feat)
                
                # 构造大Batch: Anchor + Sample2
                combined_rcs_proj = torch.cat([rcs_proj, rcs2_proj], dim=0)  # [B+B*6, 128]
                combined_tf_proj = torch.cat([tf_proj, tf2_proj], dim=0)      # [B+B*6, 128]
                
                # 拼接标签
                sample2_labels = sample2['labels'].to(labels.device)
                combined_labels = torch.cat([labels, sample2_labels], dim=0)  # [B+B*6]
                
                # 计算对比损失（在大Batch上）
                contrast_loss = self.contrast_loss(
                    combined_rcs_proj, 
                    combined_tf_proj, 
                    combined_labels
                )
            
            # 模式B: 没有Sample2 (Fallback)
            else:
                # 只在Anchor上计算对比损失
                contrast_loss = self.contrast_loss(rcs_proj, tf_proj, labels)
            
            # 总损失
            total_loss = cls_loss + self.config.contrast_loss_weight * contrast_loss
            
            return {
                'logits': logits,
                'loss': total_loss,
                'cls_loss': cls_loss,
                'contrast_loss': contrast_loss,
                'rcs_feat': rcs_global,
                'tf_feat': tf_global,
                'bottleneck_feat': bottleneck_global
            }
        
        return {'logits': logits}
    
    def get_num_params(self):
        """获取模型参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
#                           测试代码
# ============================================================================

if __name__ == '__main__':
    print("="*70)
    print("测试改进版DashFusion (保留Hard Mining + 移除Aligned投影)")
    print("="*70)
    
    # 模拟配置
    class DummyConfig:
        rcs_dim = 256
        tf_size = 256
        num_classes = 3
        hidden_dim = 128
        num_heads = 4
        num_encoder_layers = 2
        num_fusion_layers = 2
        num_bottleneck = 8
        dropout = 0.1
        temperature = 0.07
        contrast_loss_weight = 0.1
        device = 'cpu'
    
    config = DummyConfig()
    
    # 创建模型
    model = DashFusion(config)
    
    # =========================================================================
    # 测试1: 不使用Sample2
    # =========================================================================
    print("\n" + "="*70)
    print("测试1: 不使用Sample2 (Fallback模式)")
    print("="*70)
    
    batch_size = 8
    dummy_rcs = torch.randn(batch_size, 1, 256)
    dummy_tf = torch.randn(batch_size, 1, 256, 256)
    dummy_labels = torch.randint(0, 3, (batch_size,))
    
    print(f"\n输入形状:")
    print(f"  RCS: {dummy_rcs.shape}")
    print(f"  TF: {dummy_tf.shape}")
    print(f"  Labels: {dummy_labels.shape}")
    
    # Forward (不使用sample2)
    outputs = model(dummy_rcs, dummy_tf, dummy_labels, sample2=None)
    
    print(f"\n输出:")
    print(f"  Logits: {outputs['logits'].shape}")
    print(f"  Total Loss: {outputs['loss'].item():.4f}")
    print(f"  Cls Loss: {outputs['cls_loss'].item():.4f}")
    print(f"  Contrast Loss: {outputs['contrast_loss'].item():.4f}")
    
    # =========================================================================
    # 测试2: 使用Sample2 (Hard Mining模式)
    # =========================================================================
    print("\n" + "="*70)
    print("测试2: 使用Sample2 (Hard Mining模式)")
    print("="*70)
    
    # 模拟Sample2数据 (每个Anchor样本采6个)
    sample2_size = batch_size * 6
    sample2 = {
        'rcs': torch.randn(sample2_size, 1, 256),
        'tf': torch.randn(sample2_size, 1, 256, 256),
        'labels': torch.randint(0, 3, (sample2_size,))
    }
    
    print(f"\nSample2形状:")
    print(f"  RCS: {sample2['rcs'].shape}")
    print(f"  TF: {sample2['tf'].shape}")
    print(f"  Labels: {sample2['labels'].shape}")
    
    # Forward (使用sample2)
    outputs = model(dummy_rcs, dummy_tf, dummy_labels, sample2=sample2)
    
    print(f"\n输出:")
    print(f"  Logits: {outputs['logits'].shape}")
    print(f"  Total Loss: {outputs['loss'].item():.4f}")
    print(f"  Cls Loss: {outputs['cls_loss'].item():.4f}")
    print(f"  Contrast Loss: {outputs['contrast_loss'].item():.4f}")
    
    print(f"\n损失分解:")
    print(f"  Cls Loss: {outputs['cls_loss'].item():.4f} (weight: 1.0)")
    print(f"  Contrast Loss: {outputs['contrast_loss'].item():.4f} (weight: {config.contrast_loss_weight})")
    print(f"  Weighted Contrast: {outputs['contrast_loss'].item() * config.contrast_loss_weight:.4f}")
    
    # 参数量
    print(f"\n模型参数量: {model.get_num_params():,}")
    
    print("\n" + "="*70)
    print("✓ 所有测试通过! (保留Hard Mining + 移除Aligned投影)")
    print("="*70)