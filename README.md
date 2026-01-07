# DashFusion 完整代码框架梳理

## 📋 项目概述

**任务目标**：融合 RCS（雷达散射截面序列）和 TF（时频图）两种模态数据进行目标分类

**核心创新**：
1. **Hierarchical Bottleneck Fusion (HBF)** - 层次瓶颈融合机制
2. **Hard Negative Mining** - 难负样本挖掘增强对比学习
3. **Dual-Stream Alignment** - 双流时间对齐 + 语义对齐

---

## 🗂️ 文件结构与功能

### 📁 一级目录结构

```
project/
├── config.py                    # 超参数配置
├── main.py                      # 单次训练入口
├── train.py                     # 训练逻辑（核心）
├── utils.py                     # 工具函数
├── logger.py                    # 日志系统
├── experiment_configs.py        # 批量实验配置
├── run_experiments.py           # 批量实验执行器
├── session_manager.py           # 训练会话管理
│
├── dataloader/
│   └── dataset.py               # 数据加载（需实现Hard Mining接口）
│
└── model/
    ├── dashfusion.py            # 主模型（核心）
    ├── encoders.py              # RCS/TF 编码器
    ├── encoders_v2.py           # 优化版编码器（引用但缺失）
    ├── layers.py                # 注意力层、融合层
    └── MLP.py                   # 投影头、分类器
```

---

## 🔥 核心模块详解

### 1️⃣ **配置层 (config.py)**

```python
核心参数：
- 数据：rcs_dim=256, tf_size=256, num_classes=3
- 模型：hidden_dim=128, num_heads=2, num_encoder_layers=2
- 融合：num_fusion_layers=2, num_bottleneck=8
- 对比学习：temperature=0.1, contrast_loss_weight=0.05
- 训练：batch_size=30, epochs=100, learning_rate=1e-4
```

**实验配置 (experiment_configs.py)**：
- 提供多组预设配置（baseline、high_contrast、large_batch等）
- 用于批量对比实验

---

### 2️⃣ **数据层 (dataloader/dataset.py)**

**⚠️ 关键要求：Dataset 必须实现以下接口以支持 Hard Mining**

```python
class RCS_JTF_Dataset(Dataset):
    def __getitem__(self, idx):
        """必须返回4个值（而非3个）"""
        return rcs, tf, label, idx  # ← idx用于Hard Mining
    
    def update_matrix(self, all_rcs_feat, all_tf_feat):
        """Trainer调用此方法更新全局相似度矩阵"""
        # 存储所有样本的特征
        # 用于计算难负样本
        pass
    
    def sample(self, indices):
        """根据当前batch的indices，返回Sample2"""
        # 对每个样本采样6个难负样本（2个正+4个负）
        # 返回格式：
        return {
            'rcs': [B*6, 1, 256],
            'tf': [B*6, 1, 256, 256],
            'labels': [B*6]
        }
```

**数据流程**：
```
训练数据 → Dataset.__getitem__ → (rcs, tf, label, idx)
                                      ↓
                            Trainer每2轮更新相似度矩阵
                                      ↓
                            Dataset.update_matrix()
                                      ↓
                            Dataset.sample() 生成难负样本
```

---

### 3️⃣ **模型层 (model/)**

#### **3.1 主模型 (dashfusion.py)** ⭐

```python
class DashFusion(nn.Module):
    def __init__(self, config):
        # 1. 编码器
        self.rcs_encoder = OptimizedRCSEncoder(...)
        self.tf_encoder = OptimizedTFEncoder(...)
        
        # 2. 双流对齐模块
        self.dual_alignment = SimplifiedDualStreamAlignment(...)
        
        # 3. 对比学习损失
        self.contrast_loss = SimplifiedContrastiveLoss(temperature=0.07)
        
        # 4. 层次瓶颈融合
        self.hierarchical_fusion = HierarchicalBottleneckFusion(...)
        
        # 5. 分类器
        self.classifier = MultimodalClassifier(...)
        
        # 6. 分类损失（带类别权重）
        class_weights = torch.tensor([1.0, 2.5, 1.0])
        self.cls_loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    
    def forward(self, rcs, tf, labels=None, sample2=None):
        """
        核心逻辑：
        1. 编码 → 2. 对齐 → 3. 融合 → 4. 分类
        5. 计算损失（分类 + 对比学习）
        """
```

**关键改动**：
- ✅ 保留 Hard Mining（使用 sample2）
- ❌ 移除 Aligned 投影头（简化对比学习）
- ✅ 对比学习只计算 RCS ↔ TF

---

#### **3.2 双流对齐模块 (SimplifiedDualStreamAlignment)**

```python
class SimplifiedDualStreamAlignment(nn.Module):
    def temporal_alignment(self, rcs_feat, tf_feat):
        """时间对齐：使用 CrossModalAttention"""
        # TF -> RCS 注意力
        tf_to_rcs = self.tf_to_rcs(rcs_feat, tf_feat, tf_feat)
        # 残差连接
        aligned_feat = rcs_feat + tf_to_rcs
        return aligned_feat
    
    def semantic_alignment(self, rcs_feat, tf_feat):
        """语义对齐：投影到低维空间用于对比学习"""
        rcs_proj = self.rcs_projector(rcs_feat.mean(dim=1))  # [B, 128]
        tf_proj = self.tf_projector(tf_feat.mean(dim=1))      # [B, 128]
        return F.normalize(rcs_proj), F.normalize(tf_proj)
```

---

#### **3.3 对比学习损失 (SimplifiedContrastiveLoss)**

```python
class SimplifiedContrastiveLoss(nn.Module):
    """
    监督对比损失（支持Hard Mining）
    
    核心思想：
    - 同类样本拉近（正样本对）
    - 不同类样本推远（负样本对）
    - 使用Sample2增加难负样本
    """
    def forward(self, features1, features2, labels):
        # features1: RCS投影 [B, 128]
        # features2: TF投影 [B, 128]
        # B 可以是 32（Anchor）或 224（Anchor+Sample2）
        
        # 1. 拼接两个模态
        features = torch.cat([features1, features2], dim=0)  # [2B, 128]
        
        # 2. 计算相似度矩阵
        sim_matrix = torch.matmul(features, features.T) / temperature
        
        # 3. 创建正负样本mask
        mask = (labels == labels.T)  # 同类为正样本
        
        # 4. InfoNCE损失
        # log[exp(sim_pos) / sum(exp(sim_all))]
```

**Hard Mining 效果**：
- **无 Sample2**：batch_size=32 → 对比学习在 64 个样本上
- **有 Sample2**：batch_size=32 + 32×6=192 → 对比学习在 448 个样本上
- **难负样本**：4个负样本是通过相似度矩阵精心挑选的"最容易混淆"的样本

---

#### **3.4 层次瓶颈融合 (HierarchicalBottleneckFusion)**

```python
class HierarchicalBottleneckFusion(nn.Module):
    """
    核心思想：渐进式信息压缩
    Layer 0: 8个bottleneck tokens
    Layer 1: 4个bottleneck tokens  ← 减半
    Layer 2: 2个bottleneck tokens  ← 减半
    """
    def forward(self, rcs_feat, tf_feat, aligned_feat):
        # 1. 初始化bottleneck（从aligned_feat中提取）
        bottleneck = self.init_cross_attn(
            self.bottleneck_tokens,  # 可学习的query
            aligned_feat,            # key/value
            aligned_feat
        )
        
        # 2. 逐层融合
        for i, layer in enumerate(self.fusion_layers):
            # a. bottleneck自注意力
            bottleneck = self.bottleneck_transformers[i](bottleneck)
            
            # b. 信息聚合 + 特征更新
            bottleneck, rcs_feat, tf_feat = layer(
                bottleneck, rcs_feat, tf_feat
            )
        
        return bottleneck, rcs_feat, tf_feat
```

**每层的操作 (HierarchicalBottleneckFusionLayer)**：
1. **信息聚合**：bottleneck ← RCS + TF（通过 MultiCrossAttention）
2. **特征更新**：RCS ← bottleneck, TF ← bottleneck（通过 CrossAttention）

---

#### **3.5 分类器 (MultimodalClassifier)**

```python
class MultimodalClassifier(nn.Module):
    def forward(self, rcs_feat, tf_feat, bottleneck_feat):
        # 拼接三个全局特征
        fused = torch.cat([rcs_feat, tf_feat, bottleneck_feat], dim=-1)
        # [B, 128*3] → MLP → [B, num_classes]
        return self.classifier(fused)
```

---

### 4️⃣ **训练层 (train.py)** ⭐

```python
class Trainer:
    def __init__(self, model, train_loader, test_loader, config):
        # 1. 优化器（不衰减Bias和LayerNorm）
        self.optimizer = optim.AdamW(grouped_params, lr=lr)
        
        # 2. 学习率调度（Warmup + CosineAnnealing）
        self.scheduler = SequentialLR(
            [warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps]
        )
        
        # 3. 记录训练历史
        self.train_losses = []
        self.train_accs = []
        self.test_losses = []
        self.test_accs = []
    
    def update_similarity_matrix(self):
        """
        【核心逻辑】Hard Negative Mining
        
        流程：
        1. 遍历整个训练集，提取特征
        2. 调用 dataset.update_matrix() 更新相似度矩阵
        3. dataset 根据矩阵预计算难负样本
        """
        self.model.eval()
        
        # 收集全量特征
        all_rcs = torch.zeros(dataset_size, hidden_dim)
        all_tf = torch.zeros(dataset_size, hidden_dim)
        
        for batch in train_loader:
            rcs, tf, labels, indices = batch
            rcs_feat = self.rcs_encoder(rcs).mean(dim=1)
            tf_feat = self.tf_encoder(tf).mean(dim=1)
            
            all_rcs[indices] = rcs_feat
            all_tf[indices] = tf_feat
        
        # 更新Dataset的相似度矩阵
        self.train_loader.dataset.update_matrix(all_rcs, all_tf)
        
        self.model.train()
    
    def train_epoch(self, epoch):
        """训练一个Epoch"""
        for batch in train_loader:
            # 1. 解包数据（包含index）
            rcs, tf, labels, indices = batch
            
            # 2. 获取Sample2（难负样本）
            sample2 = self.train_loader.dataset.sample(indices.tolist())
            
            # 3. 前向传播（传入sample2）
            outputs = self.model(rcs, tf, labels, sample2=sample2)
            
            # 4. 反向传播
            loss = outputs['loss']
            loss.backward()
            
            # 5. 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
            optimizer.step()
            scheduler.step()
    
    def train(self):
        """主训练流程"""
        for epoch in range(1, epochs+1):
            # 【关键】每2轮更新一次相似度矩阵
            if (epoch - 1) % 2 == 0:
                self.update_similarity_matrix()
            
            # 训练
            train_loss, train_acc = self.train_epoch(epoch)
            
            # 测试
            if epoch % test_interval == 0:
                test_loss, test_acc = self.test()
                
                # 保存最佳模型
                if test_acc > best_test_acc:
                    self.save_checkpoint(epoch, is_best=True)
```

**训练策略总结**：
1. **优化器**：AdamW（不衰减Bias和Norm层）
2. **学习率**：Warmup（50步）+ CosineAnnealing
3. **梯度裁剪**：max_norm=1.0
4. **Hard Mining**：每2轮更新一次相似度矩阵
5. **损失函数**：分类损失（带类别权重）+ 对比学习损失

---

## 🔄 完整数据流

### **Forward Pass（含Hard Mining）**

```
输入：
  Anchor: rcs [B, 1, 256], tf [B, 1, 256, 256], labels [B]
  Sample2: rcs2 [B*6, 1, 256], tf2 [B*6, 1, 256, 256], labels2 [B*6]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Anchor编码
   rcs → RCSEncoder → rcs_feat [B, 256, D]
   tf  → TFEncoder  → tf_feat  [B, 256, D]

2. Anchor对齐
   ├─ 时间对齐：aligned_feat [B, 256, D]
   └─ 语义对齐：rcs_proj [B, 128], tf_proj [B, 128]

3. Anchor融合（HBF）
   (rcs_feat, tf_feat, aligned_feat) → HBF
   ├─ bottleneck [B, num_bt, D]
   ├─ rcs_fused [B, 256, D]
   └─ tf_fused [B, 256, D]

4. Anchor分类
   (rcs_global, tf_global, bottleneck_global) → Classifier
   → logits [B, num_classes]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

5. Sample2编码（对比学习用）
   rcs2 → RCSEncoder → rcs2_feat [B*6, 256, D]
   tf2  → TFEncoder  → tf2_feat  [B*6, 256, D]

6. Sample2投影
   rcs2_feat → mean → rcs_projector → rcs2_proj [B*6, 128]
   tf2_feat  → mean → tf_projector  → tf2_proj  [B*6, 128]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

7. 损失计算
   ├─ 分类损失（只用Anchor）
   │  cls_loss = CrossEntropy(logits, labels)
   │
   └─ 对比学习损失（Anchor + Sample2）
      combined_rcs_proj = [rcs_proj; rcs2_proj]  ← [B+B*6, 128]
      combined_tf_proj  = [tf_proj; tf2_proj]    ← [B+B*6, 128]
      combined_labels   = [labels; labels2]      ← [B+B*6]
      
      contrast_loss = InfoNCE(combined_rcs_proj, combined_tf_proj, combined_labels)

   total_loss = cls_loss + λ * contrast_loss
```

---

## 📊 Hard Negative Mining 详解

### **为什么需要Hard Mining？**

对比学习的效果取决于**负样本的质量**：
- **随机负样本**：很多"太简单"（模型一眼就能区分）→ 学不到东西
- **难负样本**：看起来很像但实际不同 → 强迫模型学习更细粒度的特征

### **实现原理**

```
Step 1: 构建全局相似度矩阵（每2轮更新一次）
  - 对训练集所有样本提取特征
  - 计算样本间的相似度
  - 相似度 = cosine(rcs_feat_i, rcs_feat_j) + cosine(tf_feat_i, tf_feat_j)

Step 2: 为每个样本预计算Sample2
  - 找2个同类中最相似的样本（正样本）
  - 找4个异类中最相似的样本（难负样本）
  - 存储索引，供训练时快速读取

Step 3: 训练时使用Sample2
  - DataLoader返回(rcs, tf, labels, indices)
  - 调用dataset.sample(indices)获取6个辅助样本
  - 前向传播时一起输入模型
  - 对比学习在Anchor+Sample2上计算（样本数×7）
```

### **效果对比**

| 模式 | Batch大小 | 对比学习样本数 | 效果 |
|------|----------|--------------|------|
| 无Hard Mining | 32 | 64（32×2模态） | 基线 |
| 有Hard Mining | 32 + 32×6 | 448（224×2模态） | ⬆️ 样本多+质量高 |

---

## 🚀 使用流程

### **1. 准备数据**

数据格式：
```
train_data/
  ├─ train_rcs.npy      # [N, 256]
  ├─ train_jtf.npy      # [N, 256, 256]
  └─ train_labels.npy   # [N]

test_data/
  ├─ test_rcs.npy
  ├─ test_jtf.npy
  └─ test_labels.npy
```

⚠️ **必须实现 Hard Mining 接口**：
```python
# 在 dataloader/dataset.py 中实现
class RCS_JTF_Dataset:
    def __getitem__(self, idx):
        return rcs, tf, label, idx  # ← 必须返回idx
    
    def update_matrix(self, all_rcs_feat, all_tf_feat):
        # 计算相似度矩阵并预计算Sample2
        pass
    
    def sample(self, indices):
        # 返回难负样本
        return {'rcs': ..., 'tf': ..., 'labels': ...}
```

### **2. 单次训练**

```bash
python main.py \
    --noise_level 20 \
    --epochs 100 \
    --batch_size 30 \
    --learning_rate 1e-4 \
    --contrast_loss_weight 0.05
```

### **3. 批量实验**

```bash
# 列出所有配置
python run_experiments.py --list

# 运行指定配置
python run_experiments.py \
    --experiments baseline high_contrast large_batch

# 运行所有配置
python run_experiments.py

# 结果保存在 ./experiments/session_YYYYMMDD_HHMMSS/
```

### **4. 查看历史实验**

```bash
# 列出所有训练sessions
python session_manager.py list

# 查看某个session详情
python session_manager.py view 20250107_123456

# 对比两个sessions
python session_manager.py compare 20250107_123456 20250107_143210

# 显示最佳session
python session_manager.py best

# 清理旧sessions（保留最近5个）
python session_manager.py clean --keep 5 --no-dry-run
```

---

## 🎯 关键技术总结

### **1. Hierarchical Bottleneck Fusion**
- 渐进式压缩：8→4→2个bottleneck tokens
- 双向交互：bottleneck←→RCS, bottleneck←→TF
- 多尺度融合：每层提取不同粒度的信息

### **2. Hard Negative Mining**
- 动态更新：每2轮重新计算相似度矩阵
- 质量保证：选择"最容易混淆"的负样本
- 效率提升：预计算索引，训练时快速读取

### **3. Dual-Stream Alignment**
- 时间对齐：CrossModalAttention统一时序表示
- 语义对齐：投影到低维空间进行对比学习
- 简化策略：只计算RCS↔TF（移除Aligned投影）

### **4. 训练策略**
- 优化器：AdamW（选择性weight decay）
- 学习率：Warmup + CosineAnnealing
- 损失函数：分类损失（带类别权重）+ 对比学习损失
- 正则化：Dropout + Gradient Clipping

---

## 📝 注意事项

### **⚠️ 必须实现的接口**

1. **Dataset必须返回4个值**：`(rcs, tf, label, idx)`
2. **Dataset必须实现**：
   - `update_matrix(all_rcs_feat, all_tf_feat)`
   - `sample(indices) -> {'rcs', 'tf', 'labels'}`

### **⚠️ 缺失的文件**

- `encoders_v2.py` - 在 `dashfusion.py` 中被引用但不存在
  - 解决方案：使用 `encoders.py` 中的 `RCSEncoder` 和 `TFEncoder`
  - 或创建优化版编码器

### **⚠️ 类别不平衡处理**

```python
# 方法1: 类别权重（当前使用）
class_weights = torch.tensor([1.0, 2.5, 1.0])
cls_loss_fn = nn.CrossEntropyLoss(weight=class_weights)

# 方法2: Focal Loss
cls_loss_fn = FocalLoss(alpha=1, gamma=2.0)
```

---

## 📈 性能优化建议

### **1. 数据增强**
```python
# 在 dataloader/dataset.py 中实现
- RCS: 噪声注入、时移
- TF: 噪声注入、频谱增强
```

### **2. 超参数调优**
```python
关键参数：
- contrast_loss_weight: [0.01, 0.05, 0.1, 0.2]
- temperature: [0.05, 0.07, 0.1]
- num_bottleneck: [4, 8, 16]
- learning_rate: [5e-5, 1e-4, 2e-4]
```

### **3. 模型压缩**
```python
# 减少参数量
- num_encoder_layers: 2 → 1
- num_fusion_layers: 2 → 1
- hidden_dim: 128 → 64
- num_bottleneck: 8 → 4
```

---

## 🔧 常见问题

### **Q1: 如何实现Hard Mining的Dataset？**

参考原文代码中的 `Sample2` 采样逻辑，需要：
1. 存储全局相似度矩阵
2. 预计算每个样本的6个邻居（2正+4负）
3. 在 `sample()` 方法中快速返回

### **Q2: 为什么需要返回idx？**

`idx` 用于将DataLoader打乱后的样本映射回原始位置，以便：
1. 在相似度矩阵中找到对应位置
2. 正确地聚合特征

### **Q3: 训练慢怎么办？**

1. 减少 `update_similarity_matrix` 频率（改为每5轮）
2. 减少 Sample2 数量（6→3）
3. 减少 batch_size 或模型深度

---

## 📚 参考资源

- **论文**：DashFusion（查找原文获取详细算法）
- **数据集**：RCS + Time-Frequency 雷达信号数据
- **相关技术**：
  - Supervised Contrastive Learning
  - Hard Negative Mining
  - Hierarchical Fusion

---

**文档生成时间**：2025-01-07  
**框架版本**：DashFusion with Hard Mining
