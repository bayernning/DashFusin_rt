"""
实验配置文件 - 多组超参数对比实验
"""

EXPERIMENT_CONFIGS = [
    # {
    #     'name': 'baseline',
    #     'description': '基线配置 - DashFusion原文推荐参数',
    #     'params': {
    #         'learning_rate': 1e-4,
    #         'batch_size': 30,
    #         'epochs': 100,
    #         'num_bottleneck': 8,
    #         'num_fusion_layers': 2,
    #         'contrast_loss_weight': 0.01,
    #         'temperature': 0.1,
    #         'dropout': 0.3,
    #         'warmup_steps': 50,
    #         'weight_decay': 1e-4,
    #     }
    # },
    
    # {
    #     'name': 'high_contrast',
    #     'description': '高对比学习权重 - 强化模态对齐',
    #     'params': {
    #         'learning_rate': 1e-4,
    #         'batch_size': 30,
    #         'epochs':  100,
    #         'num_bottleneck': 8,
    #         'num_fusion_layers': 2,
    #         'contrast_loss_weight': 0.02,  # ← 提高
    #         'temperature': 0.1,
    #         'dropout': 0.3,
    #         'warmup_steps': 50,
    #         'weight_decay': 1e-4,
    #     }
    # },
    
    # {
    #     'name': 'large_batch',
    #     'description': '大批次训练 - 更稳定的梯度',
    #     'params': {
    #         'learning_rate': 1e-4,  # ← 大batch需要更大lr
    #         'batch_size': 60,       # ← 增大
    #         'epochs': 100,
    #         'num_bottleneck': 8,
    #         'num_fusion_layers': 2,
    #         'contrast_loss_weight': 0.05,
    #         'temperature': 0.1,
    #         'dropout': 0.3,
    #         'warmup_steps': 100,    # ← 增加warmup
    #         'weight_decay': 1e-4,
    #     }
    # },
    
    # {
    #     'name': 'deep_fusion',
    #     'description': '深层融合 - 更强的特征提取',
    #     'params': {
    #         'learning_rate': 5e-5,  # ← 更深的网络用小lr
    #         'batch_size': 30,
    #         'epochs': 100,
    #         'num_bottleneck': 8,
    #         'num_fusion_layers': 3,  # ← 增加层数
    #         'contrast_loss_weight': 0.02,
    #         'temperature': 0.1,
    #         'dropout': 0.2,          # ← 增加正则化
    #         'warmup_steps': 50,
    #         'weight_decay': 5e-5,
    #     }
    # },
    
    # {
    #     'name': 'more_bottleneck',
    #     'description': '更多瓶颈tokens - 保留更多信息',
    #     'params': {
    #         'learning_rate': 1e-4,
    #         'batch_size': 30,
    #         'epochs': 100,
    #         'num_bottleneck': 16,    # ← 增加
    #         'num_fusion_layers': 2,
    #         'contrast_loss_weight': 0.01,
    #         'temperature': 0.1,
    #         'dropout': 0.3,
    #         'warmup_steps': 50,
    #         'weight_decay': 1e-4,
    #     }
    # },
    
    # {
    #     'name': 'low_temp',
    #     'description': '低温度对比学习 - 更严格的样本区分',
    #     'params': {
    #         'learning_rate': 1e-4,
    #         'batch_size': 30,
    #         'epochs': 100,
    #         'num_bottleneck': 8,
    #         'num_fusion_layers': 2,
    #         'contrast_loss_weight': 0.015,
    #         'temperature': 0.05,     # ← 降低温度
    #         'dropout': 0.3,
    #         'warmup_steps': 50,
    #         'weight_decay': 1e-4,
    #     }
    # },
    
    # {
    #     'name': 'conservative',
    #     'description': '保守配置 - 防止过拟合（小数据集推荐）',
    #     'params': {
    #         'learning_rate': 5e-5,   # ← 小lr
    #         'batch_size': 16,        # ← 小batch
    #         'epochs': 100,
    #         'num_bottleneck': 8,
    #         'num_fusion_layers': 2,
    #         'contrast_loss_weight': 0.01,  # ← 弱对比
    #         'temperature': 0.2,      # ← 高温度
    #         'dropout': 0.1,          # ← 强dropout
    #         'warmup_steps': 30,
    #         'weight_decay': 5e-3,    # ← 强L2正则
    #     }
    # },
    
    # {
    #     'name': 'aggressive',
    #     'description': '激进配置 - 快速收敛（大数据集推荐）',
    #     'params': {
    #         'learning_rate': 2e-4,   # ← 大lr
    #         'batch_size': 30,        # ← 大batch
    #         'epochs': 100,
    #         'num_bottleneck': 16,    # ← 大容量
    #         'num_fusion_layers': 3,  # ← 深层
    #         'contrast_loss_weight': 0.1,  # ← 强对比
    #         'temperature': 0.1,
    #         'dropout': 0.2,          # ← 弱dropout
    #         'warmup_steps': 100,
    #         'weight_decay': 1e-4,    # ← 弱L2
    #     }
    # },
    # {
    #     'name': 'claudesuggest',
    #     'description': '激进配置 - 快速收敛（大数据集推荐）',
    #     'params': {
    #         'learning_rate': 1e-4,   # ← 大lr
    #         'batch_size': 30,        # ← 大batch
    #         'epochs': 100,
    #         'num_bottleneck': 4,    
    #         'num_fusion_layers': 1,  
    #         'contrast_loss_weight': 0.1,  
    #         'temperature': 0.1,
    #         'dropout': 0.2,          
    #         'warmup_steps': 100,
    #         'weight_decay': 1e-4,   
    #         'hidden_dim':64,

    #     }
    # },
    {
        'name': 'claudesuggest',
        'description': '激进配置 - 快速收敛（大数据集推荐）',
        'params': {
            'learning_rate': 1e-4,   # ← 大lr
            'batch_size': 30,        # ← 大batch
            'epochs': 100,
            'num_bottleneck': 4,    
            'num_fusion_layers': 1,  
            'contrast_loss_weight': 0.2,  
            'temperature': 0.1,
            'dropout': 0.2,          
            'warmup_steps': 100,
            'weight_decay': 1e-4,   
            'hidden_dim':64,
            'num_fusion_layers':1,
            'num_encoder_layers':1
        }
    },
    
]


def get_experiment_config(name):
    """根据名称获取实验配置"""
    for config in EXPERIMENT_CONFIGS:
        if config['name'] == name:
            return config
    raise ValueError(f"未找到实验配置: {name}")


def list_experiments():
    """列出所有实验配置"""
    print("\n" + "="*80)
    print("可用的实验配置:")
    print("="*80)
    for i, config in enumerate(EXPERIMENT_CONFIGS, 1):
        print(f"\n{i}. {config['name']}")
        print(f"   描述: {config['description']}")
        print(f"   关键参数:")
        params = config['params']
        print(f"     - 学习率: {params['learning_rate']}")
        print(f"     - 批次大小: {params['batch_size']}")
        print(f"     - 对比学习权重: {params['contrast_loss_weight']}")
        print(f"     - 温度: {params['temperature']}")
        print(f"     - 瓶颈数量: {params['num_bottleneck']}")
        print(f"     - 融合层数: {params['num_fusion_layers']}")
    print("="*80 + "\n")


if __name__ == '__main__':
    list_experiments()
