"""
配置文件 - DashFusion for RCS & JTF
"""
import argparse
import torch

def get_config():
    parser = argparse.ArgumentParser()
    
    # 数据相关
    parser.add_argument('--dataset', type=str, default='rcs_tf', help='数据集名称')
    parser.add_argument('--train_data_dir', type=str, default='./train_data/', help='训练数据路径')
    parser.add_argument('--test_data_dir', type=str, default='./test_data/', help='测试数据路径')
    parser.add_argument('--noise_level', type=int, default=0, help='噪声级别(dB), 如0, 5, 10等')
    parser.add_argument('--rcs_dim', type=int, default=256, help='RCS序列长度')
    parser.add_argument('--tf_size', type=int, default=256, help='TF图像大小')
    parser.add_argument('--num_classes', type=int, default=10, help='分类类别数')
    
    # 模型相关
    parser.add_argument('--hidden_dim', type=int, default=128, help='隐藏层维度')
    parser.add_argument('--num_heads', type=int, default=4, help='注意力头数')
    parser.add_argument('--num_encoder_layers', type=int, default=2, help='编码器层数')
    parser.add_argument('--num_fusion_layers', type=int, default=2, help='融合层数')
    parser.add_argument('--num_bottleneck', type=int, default=8, help='初始瓶颈token数量')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout率')
    
    # 对比学习相关
    parser.add_argument('--temperature', type=float, default=0.5, help='对比学习温度参数')
    parser.add_argument('--contrast_loss_weight', type=float, default=0.2, help='对比学习损失权重')
    
    # 训练相关
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='权重衰减')
    parser.add_argument('--warmup_steps', type=int, default=100, help='预热步数')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='梯度裁剪')
    parser.add_argument('--test_interval', type=int, default=10, help='测试间隔(每N轮测试一次)')
    
    # 设备相关
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载线程数')
    
    # 日志和保存相关
    parser.add_argument('--log_dir', type=str, default='./log/', help='日志保存路径')
    parser.add_argument('--ckpt_dir', type=str, default='./ckpt/', help='模型保存路径')
    parser.add_argument('--result_dir', type=str, default='./result/', help='结果保存路径')
    parser.add_argument('--save_interval', type=int, default=10, help='保存间隔')
    parser.add_argument('--log_interval', type=int, default=10, help='日志打印间隔')
    
    args = parser.parse_args()
    return args