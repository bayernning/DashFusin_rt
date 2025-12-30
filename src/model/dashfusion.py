"""
完整的DashFusion模型 for RCS & JTF
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.encoders import RCSEncoder, TFEncoder
from model.layers import CrossModalAttention, HierarchicalBottleneckFusion
from model.MLP import DualProjector, MultimodalClassifier


class DualStreamAlignment(nn.Module):
    """双流对齐模块: 时间对齐 + 语义对齐"""
    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        # 时间对齐: 跨模态注意力
        self.tf_to_rcs = CrossModalAttention(hidden_dim, num_heads, dropout)
        
        # 语义对齐: 使用MLP中的投影头
        from model.MLP import Projector
        self.rcs_projector = Projector(hidden_dim, output_dim=128, dropout=dropout)
        self.tf_projector = Projector(hidden_dim, output_dim=128, dropout=dropout)
        
        self.norm = nn.LayerNorm(hidden_dim)
        
    def temporal_alignment(self, rcs_feat, tf_feat):
        """
        时间对齐: 以RCS为锚点，将TF对齐到RCS
        rcs_feat: [batch, seq_len_rcs, hidden_dim]
        tf_feat: [batch, seq_len_tf, hidden_dim]
        """
        # TF -> RCS
        tf_to_rcs = self.tf_to_rcs(rcs_feat, tf_feat, tf_feat)
        
        # 融合: RCS + aligned_TF
        aligned_feat = self.norm(rcs_feat + tf_to_rcs)
        
        return aligned_feat
    
    def semantic_alignment(self, rcs_feat, tf_feat):
        """
        语义对齐: 获取全局特征并投影到对比学习空间
        """
        # 全局平均池化
        rcs_global = rcs_feat.mean(dim=1)  # [batch, hidden_dim]
        tf_global = tf_feat.mean(dim=1)  # [batch, hidden_dim]
        
        # 投影
        rcs_proj = self.rcs_projector(rcs_global)
        tf_proj = self.tf_projector(tf_global)
        
        return rcs_proj, tf_proj
    
    def forward(self, rcs_feat, tf_feat):
        # 时间对齐
        aligned_feat = self.temporal_alignment(rcs_feat, tf_feat)
        
        # 语义对齐
        rcs_proj, tf_proj = self.semantic_alignment(rcs_feat, tf_feat)
        
        return aligned_feat, rcs_proj, tf_proj


class SupervisedContrastiveLoss(nn.Module):
    """监督对比学习损失 (NT-Xent)"""
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, features1, features2, labels):
        """
        features1, features2: [batch, proj_dim] - 两个模态的投影特征
        labels: [batch] - 样本标签
        """
        device = features1.device
        batch_size = features1.shape[0]
        
        # 拼接两个模态的特征
        features = torch.cat([features1, features2], dim=0)  # [2*batch, proj_dim]
        labels = labels.repeat(2)  # [2*batch]
        
        # 计算相似度矩阵
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # 创建mask: 同类为正样本
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        
        # 去除对角线 (自己和自己)
        logits_mask = torch.ones_like(mask).fill_diagonal_(0)
        mask = mask * logits_mask
        
        # 计算log_prob
        exp_logits = torch.exp(similarity_matrix) * logits_mask
        log_prob = similarity_matrix - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-9)
        
        # 计算平均log-likelihood
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-9)
        
        # 损失
        loss = -mean_log_prob_pos.mean()
        
        return loss


class DashFusion(nn.Module):
    """
    完整的DashFusion模型
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 1. 模态编码
        self.rcs_encoder = RCSEncoder(
            rcs_dim=config.rcs_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_encoder_layers,
            num_heads=config.num_heads,
            dropout=config.dropout
        )
        
        self.tf_encoder = TFEncoder(
            hidden_dim=config.hidden_dim,
            num_layers=config.num_encoder_layers,
            num_heads=config.num_heads,
            dropout=config.dropout
        )
        
        # 2. 双流对齐
        self.dual_alignment = DualStreamAlignment(
            hidden_dim=config.hidden_dim,
            num_heads=config.num_heads,
            dropout=config.dropout
        )
        
        # 3. 监督对比学习
        self.contrast_loss = SupervisedContrastiveLoss(
            temperature=config.temperature
        )
        
        # 4. 层次瓶颈融合
        self.hierarchical_fusion = HierarchicalBottleneckFusion(
            hidden_dim=config.hidden_dim,
            num_bottleneck=config.num_bottleneck,
            num_layers=config.num_fusion_layers,
            num_heads=config.num_heads,
            dropout=config.dropout
        )
        
        # 5. 分类器 - 使用MultimodalClassifier
        self.classifier = MultimodalClassifier(
            rcs_dim=config.hidden_dim,
            jtf_dim=config.hidden_dim,  # TF特征维度
            bottleneck_dim=config.hidden_dim,
            hidden_dims=[config.hidden_dim * 2, config.hidden_dim],
            num_classes=config.num_classes,
            dropout=config.dropout
        )
        
    def forward(self, rcs, tf, labels=None):
        """
        rcs: [batch, 1, 256]
        tf: [batch, 1, H, W]
        labels: [batch] (optional)
        """
        # 1. 模态编码
        rcs_feat = self.rcs_encoder(rcs)      # [batch, 256, hidden_dim]
        tf_feat = self.tf_encoder(tf)         # [batch, seq_len, hidden_dim]
        
        # 2. 双流对齐
        aligned_feat, rcs_proj, tf_proj = self.dual_alignment(rcs_feat, tf_feat)
        
        # 3. 层次瓶颈融合
        bottleneck, rcs_fused, tf_fused = self.hierarchical_fusion(
            rcs_feat, tf_feat, aligned_feat
        )
        
        # 4. 全局特征提取
        rcs_global = rcs_fused.mean(dim=1)      # [batch, hidden_dim]
        tf_global = tf_fused.mean(dim=1)        # [batch, hidden_dim]
        bottleneck_global = bottleneck.mean(dim=1)  # [batch, hidden_dim]
        
        # 5. 分类 - 使用MultimodalClassifier
        logits = self.classifier(rcs_global, tf_global, bottleneck_global)
        
        # 计算损失
        if labels is not None:
            # 分类损失
            cls_loss = F.cross_entropy(logits, labels)
            
            # 对比学习损失
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