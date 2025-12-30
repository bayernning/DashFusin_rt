"""
工具函数
"""
import torch
import numpy as np
import random
import os


def set_seed(seed):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")


def count_parameters(model):
    """统计模型参数"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    return total, trainable


def save_config(config, save_path):
    """保存配置"""
    import json
    config_dict = vars(config)
    # 转换不可序列化的类型
    for key, value in config_dict.items():
        if isinstance(value, torch.device):
            config_dict[key] = str(value)
    
    with open(save_path, 'w') as f:
        json.dump(config_dict, f, indent=4)
    print(f"Config saved to {save_path}")


def load_config(load_path):
    """加载配置"""
    import json
    with open(load_path, 'r') as f:
        config_dict = json.load(f)
    print(f"Config loaded from {load_path}")
    return config_dict


class AverageMeter:
    """计算和存储平均值"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """计算top-k准确率"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def print_model_summary(model, input_rcs_shape=(16, 1, 256), input_jtf_shape=(16, 1, 256, 256)):
    """打印模型摘要"""
    print("\n" + "="*60)
    print("Model Summary")
    print("="*60)
    
    # 统计参数
    total_params = 0
    trainable_params = 0
    
    print(f"\n{'Layer':<40} {'Parameters':>15}")
    print("-"*60)
    
    for name, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params
        if param.requires_grad:
            trainable_params += num_params
        
        # 只显示主要层
        if any(x in name for x in ['encoder', 'alignment', 'fusion', 'classifier']):
            print(f"{name:<40} {num_params:>15,}")
    
    print("-"*60)
    print(f"{'Total Parameters':<40} {total_params:>15,}")
    print(f"{'Trainable Parameters':<40} {trainable_params:>15,}")
    print(f"{'Non-trainable Parameters':<40} {total_params - trainable_params:>15,}")
    print("="*60 + "\n")
    
    # 测试前向传播
    device = next(model.parameters()).device
    try:
        dummy_rcs = torch.randn(*input_rcs_shape).to(device)
        dummy_jtf = torch.randn(*input_jtf_shape).to(device)
        
        model.eval()
        with torch.no_grad():
            outputs = model(dummy_rcs, dummy_jtf)
        
        print(f"Forward pass test successful!")
        print(f"Input RCS shape: {input_rcs_shape}")
        print(f"Input JTF shape: {input_jtf_shape}")
        print(f"Output logits shape: {outputs['logits'].shape}")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"Forward pass test failed: {e}")


def visualize_training_history(history_path, save_path=None):
    """可视化训练历史"""
    try:
        import matplotlib.pyplot as plt
        
        history = np.load(history_path, allow_pickle=True).item()
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 损失曲线
        epochs = range(1, len(history['train_losses']) + 1)
        axes[0].plot(epochs, history['train_losses'], 'b-', label='Train Loss', linewidth=2)
        
        # 测试损失（在特定epoch）
        if 'test_epochs' in history and len(history['test_epochs']) > 0:
            axes[0].plot(history['test_epochs'], history['test_losses'], 
                        'r-o', label='Test Loss', linewidth=2, markersize=6)
        
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].set_title('Training and Test Loss', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        # 准确率曲线
        axes[1].plot(epochs, history['train_accs'], 'b-', label='Train Acc', linewidth=2)
        
        # 测试准确率（在特定epoch）
        if 'test_epochs' in history and len(history['test_epochs']) > 0:
            axes[1].plot(history['test_epochs'], history['test_accs'], 
                        'r-o', label='Test Acc', linewidth=2, markersize=6)
            # 标记最佳点
            best_idx = np.argmax(history['test_accs'])
            best_epoch = history['test_epochs'][best_idx]
            best_acc = history['test_accs'][best_idx]
            axes[1].axhline(y=best_acc, color='g', linestyle='--', 
                          label=f'Best: {best_acc:.2f}% @Epoch{best_epoch}', linewidth=1.5)
            axes[1].plot(best_epoch, best_acc, 'g*', markersize=15)
        
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Accuracy (%)', fontsize=12)
        axes[1].set_title('Training and Test Accuracy', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"训练历史图已保存到: {save_path}")
        else:
            plt.show()
        
        plt.close()
        
    except ImportError:
        print("matplotlib未安装，跳过可视化")
    except Exception as e:
        print(f"可视化训练历史时出错: {e}")


if __name__ == '__main__':
    # 测试工具函数
    set_seed(42)
    
    # 测试AverageMeter
    meter = AverageMeter()
    for i in range(10):
        meter.update(i)
    print(f"Average: {meter.avg}")