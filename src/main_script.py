"""
主程序 - DashFusion for RCS & JTF
"""
import torch
import argparse
import os
import sys

from config import get_config
from model.dashfusion import DashFusion
from dataloader import get_dataloader
from train import Trainer, test, load_checkpoint
from utils import set_seed, save_config, print_model_summary, visualize_training_history


def main():
    # 获取配置
    config = get_config()
    
    # 设置随机种子
    set_seed(config.seed)
    
    # 设置设备
    device = torch.device(config.device)
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # 保存配置
    save_config(config, os.path.join(config.log_dir, 'config.json'))
    
    # 创建数据加载器
    print("\n" + "="*60)
    print("Loading Data")
    print("="*60)
    train_loader = get_dataloader(config, split='train')
    val_loader = get_dataloader(config, split='val')
    test_loader = get_dataloader(config, split='test')
    
    # 创建模型
    print("\n" + "="*60)
    print("Building Model")
    print("="*60)
    model = DashFusion(config).to(device)
    
    # 打印模型摘要
    print_model_summary(model)
    
    # 训练
    print("\n" + "="*60)
    print("Starting Training")
    print("="*60)
    trainer = Trainer(model, train_loader, val_loader, config)
    best_val_acc = trainer.train()
    
    # 可视化训练历史
    history_path = os.path.join(config.result_dir, 'history.npy')
    plot_path = os.path.join(config.result_dir, 'training_history.png')
    visualize_training_history(history_path, plot_path)
    
    # 加载最佳模型进行测试
    print("\n" + "="*60)
    print("Testing Best Model")
    print("="*60)
    best_model_path = os.path.join(config.ckpt_dir, 'best.pth')
    if os.path.exists(best_model_path):
        model = load_checkpoint(model, best_model_path, device)
        test_acc = test(model, test_loader, config)
        
        # 保存最终结果
        final_results = {
            'best_val_acc': best_val_acc,
            'test_acc': test_acc,
            'best_epoch': trainer.best_epoch
        }
        
        # 打印最终结果
        print("\n" + "="*60)
        print("Final Results")
        print("="*60)
        print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
        print(f"Test Accuracy: {test_acc:.2f}%")
        print(f"Best Epoch: {trainer.best_epoch}")
        print("="*60 + "\n")
        
        # 保存结果到文件
        with open(os.path.join(config.result_dir, 'final_results.txt'), 'w') as f:
            f.write(f"Best Validation Accuracy: {best_val_acc:.2f}%\n")
            f.write(f"Test Accuracy: {test_acc:.2f}%\n")
            f.write(f"Best Epoch: {trainer.best_epoch}\n")
    
    print("Training and testing completed!")


if __name__ == '__main__':
    main()