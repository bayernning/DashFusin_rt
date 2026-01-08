"""
主程序 - DashFusion for RCS & TF
"""
import torch
import os
import sys

from config import get_config
from model.dashfusion import DashFusion
from dataloader.dataset_12 import get_dataloader, check_data_format
from train import Trainer, final_test, load_checkpoint
from utils import set_seed, save_config, print_model_summary, visualize_training_history

def main():
    # 1. 获取配置
    config = get_config()
    
    # 2. 基础设置
    set_seed(config.seed)
    device = torch.device(config.device)
    
    # 3. 创建目录
    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(config.ckpt_dir, exist_ok=True)
    os.makedirs(config.result_dir, exist_ok=True)
    
    # 4. 打印环境信息
    print(f"\n{'='*60}")
    print(f"环境配置")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Noise Level: {config.noise_level} dB")
    
    # 5. 检查数据
    if not check_data_format(config):
        print("错误：数据格式检查失败！")
        return
        
    # 6. 加载数据
    train_loader = get_dataloader(config, split='train')
    test_loader = get_dataloader(config, split='test')
    
    # 7. 构建模型
    model = DashFusion(config).to(device)
    print_model_summary(model)
    
    # 8. 训练
    # 注意：Logger 会在 Trainer 内部初始化
    trainer = Trainer(model, train_loader, test_loader, config)
    best_test_acc = trainer.train()
    
    # 9. 可视化
    print(f"\n正在生成训练曲线...")
    history_path = os.path.join(config.result_dir, 'history.npy')
    plot_path = os.path.join(config.result_dir, 'training_history.png')
    if os.path.exists(history_path):
        visualize_training_history(history_path, plot_path)
    
    # 10. 最终测试 (使用最佳权重)
    print(f"\n{'='*60}")
    print(f"执行最终测试 (加载最佳权重)")
    print(f"{'='*60}")
    
    # train.py 现在保证会生成 best.pth
    best_model_path = os.path.join(config.ckpt_dir, 'best.pth')
    
    if os.path.exists(best_model_path):
        # 重新加载模型架构以防万一，或者直接load权重
        model = load_checkpoint(model, best_model_path, device)
        final_acc = final_test(model, test_loader, config)
        
        # 写入最终结果摘要
        summary_path = os.path.join(config.result_dir, 'summary.txt')
        with open(summary_path, 'w') as f:
            f.write(f"Best Val Acc (during train): {best_test_acc:.2f}%\n")
            f.write(f"Final Test Acc (best model): {final_acc:.2f}%\n")
        print(f"结果摘要已保存至: {summary_path}")
    else:
        print(f"警告: 未找到最佳模型文件 {best_model_path}，跳过最终测试。")

if __name__ == '__main__':
    main()