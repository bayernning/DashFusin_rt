# DashFusion for RCS & JTF

基于论文 "DashFusion: Dual-stream Alignment with Hierarchical Bottleneck Fusion for Multimodal Sentiment Analysis" 的RCS序列和JTF时频图双模态分析实现。

## 📁 项目结构

```
DashFusion_RCS_JTF/
├── config.py              # 配置文件
├── layers.py              # 核心层定义(Attention, HBF等)
├── encoders.py            # RCS和JTF编码器
├── dashfusion.py          # 完整DashFusion模型
├── dataloader.py          # 数据加载器
├── train.py               # 训练脚本
├── utils.py               # 工具函数
├── main.py                # 主程序
├── requirements.txt       # 依赖包
├── dataset/               # 数据目录
│   ├── train_rcs.npy
│   ├── train_jtf.npy
│   ├── train_labels.npy
│   ├── val_rcs.npy
│   ├── val_jtf.npy
│   ├── val_labels.npy
│   ├── test_rcs.npy
│   ├── test_jtf.npy
│   └── test_labels.npy
├── log/                   # 日志目录
├── ckpt/                  # 模型检查点目录
└── result/                # 结果目录
```

## 🚀 快速开始

### 1. 环境安装

```bash
# 创建conda环境
conda create -n dashfusion python=3.9
conda activate dashfusion

# 安装依赖
pip install -r requirements.txt
```

### 2. 数据准备

#### 数据格式
- **RCS数据**: shape为 `[N, 256]` 的numpy数组，保存为`.npy`文件
- **JTF数据**: shape为 `[N, 256, 256]` 的numpy数组，保存为`.npy`文件  
- **标签数据**: shape为 `[N]` 的numpy数组，保存为`.npy`文件

#### 数据组织
将数据按如下命名放入 `dataset/` 目录：
```
dataset/
├── train_rcs.npy
├── train_jtf.npy
├── train_labels.npy
├── val_rcs.npy
├── val_jtf.npy
├── val_labels.npy
├── test_rcs.npy
├── test_jtf.npy
└── test_labels.npy
```

**注意**: 如果数据不存在，程序会自动创建虚拟数据用于测试。

### 3. 训练模型

```bash
python main.py
```

可以通过命令行参数修改配置：

```bash
python main.py --batch_size 32 --learning_rate 1e-4 --epochs 50
```

### 4. 测试单个模块

```bash
# 测试数据加载器
python dataloader.py

# 测试工具函数
python utils.py
```

## ⚙️ 主要配置参数

在 `config.py` 中可以修改以下参数：

### 数据相关
- `rcs_dim`: RCS序列长度 (默认: 256)
- `jtf_size`: JTF图像大小 (默认: 256)
- `num_classes`: 分类类别数 (默认: 10)

### 模型相关
- `hidden_dim`: 隐藏层维度 (默认: 128)
- `num_heads`: 注意力头数 (默认: 4)
- `num_encoder_layers`: 编码器层数 (默认: 2)
- `num_fusion_layers`: 融合层数 (默认: 2)
- `num_bottleneck`: 初始瓶颈token数量 (默认: 8)

### 训练相关
- `batch_size`: 批次大小 (默认: 16)
- `epochs`: 训练轮数 (默认: 100)
- `learning_rate`: 学习率 (默认: 5e-5)
- `contrast_loss_weight`: 对比学习损失权重 (默认: 0.2)

## 📊 模型架构

### 1. 双流对齐 (Dual-stream Alignment)
- **时间对齐**: 使用跨模态注意力将JTF特征对齐到RCS序列
- **语义对齐**: 通过对比学习在特征空间中拉近同类样本

### 2. 监督对比学习 (Supervised Contrastive Learning)
- 利用标签信息增强特征判别能力
- 使用NT-Xent损失

### 3. 层次瓶颈融合 (Hierarchical Bottleneck Fusion)
- 渐进式压缩多模态信息
- 每层瓶颈token数量减半
- 双向信息流：收集信息 → 更新特征

## 📈 输入输出

### 输入
```python
rcs: [batch_size, 1, 256]      # RCS时域序列
jtf: [batch_size, 1, 256, 256] # JTF时频图
labels: [batch_size]            # 标签 (可选)
```

### 输出
```python
outputs = {
    'logits': [batch_size, num_classes],  # 分类logits
    'loss': scalar,                        # 总损失
    'cls_loss': scalar,                    # 分类损失
    'contrast_loss': scalar,               # 对比学习损失
    'rcs_feat': [batch_size, hidden_dim],  # RCS全局特征
    'jtf_feat': [batch_size, hidden_dim],  # JTF全局特征
    'bottleneck_feat': [batch_size, hidden_dim]  # 瓶颈全局特征
}
```

## 🔧 自定义数据适配

如果你的数据格式不同，需要修改 `dataloader.py` 中的 `RCS_JTF_Dataset` 类：

```python
class RCS_JTF_Dataset(Dataset):
    def __init__(self, rcs_path, jtf_path, label_path, transform=None):
        # 加载你的数据
        self.rcs_data = load_your_rcs_data(rcs_path)
        self.jtf_data = load_your_jtf_data(jtf_path)
        self.labels = load_your_labels(label_path)
    
    def __getitem__(self, idx):
        # 返回 [1, 256], [1, 256, 256], label
        return rcs, jtf, label
```

## 📝 训练技巧

1. **学习率调整**: 使用warmup + cosine annealing策略
2. **梯度裁剪**: 默认裁剪到1.0，防止梯度爆炸
3. **数据增强**: 对RCS添加噪声和时移，对JTF添加噪声
4. **早停**: 监控验证准确率，保存最佳模型

## 🎯 性能优化建议

1. **批次大小**: 根据GPU显存调整，建议16-32
2. **隐藏维度**: 增大可提升性能但增加计算量
3. **瓶颈数量**: 8-16较合适，太大会引入冗余
4. **融合层数**: 2-3层即可，过多会过拟合

## 📚 参考文献

```bibtex
@ARTICLE{wen2025dashfusion,
  author={Wen, Yuhua and Li, Qifei and Zhou, Yingying and Gao, Yingming 
          and Wen, Zhengqi and Tao, Jianhua and Li, Ya},
  journal={IEEE Transactions on Neural Networks and Learning Systems}, 
  title={DashFusion: Dual-Stream Alignment With Hierarchical Bottleneck 
         Fusion for Multimodal Sentiment Analysis}, 
  year={2025},
  volume={36},
  number={10},
  pages={17941-17952},
  doi={10.1109/TNNLS.2025.3578618}
}
```

## 📧 联系方式

如有问题，请提issue或联系开发者。

## 📄 许可证

本项目遵循MIT许可证。
