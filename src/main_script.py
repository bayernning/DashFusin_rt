"""
主程序 - DashFusion for RCS & TF
训练+测试模式，无验证集
"""
import torch
import argparse
import os
import sys

from config import get_config
from dashfusion import DashFusion
from dataloader import get_dataloader, check_data_format
from train import Trainer, final_test, load_checkpoint
from utils import set_seed, save_config, print_model_summary, visualize_training_history


def main():
    # 获取配置
    config = get_config()
    
    # 设置随机种子
    set_seed(config.seed)
    
    # 设置设备
    device = torch.device(config.device)
    print(f"使用设备: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # 创建保存目录
    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(config.ckpt_dir, exist_ok=True)
    os.makedirs(config.result_dir, exist_ok=True)
    
    # 保存配置
    save_config(config, os.path.join(config.log_dir, 'config.json'))
    
    # 检查数据格式
    print("\n" + "="*60)
    print("检查数据格式")
    print("="*60)
    if not check_data_format(config):
        print("数据格式检查失败，请检查数据文件！")
        return
    
    # 创建数据加载器
    print("\n" + "="*60)
    print("加载数据")
    print("="*60)
    print(f"噪声级别: {config.noise_level}dB")
    train_loader = get_dataloader(config, split='train')
    test_loader = get_dataloader(config, split='test')
    
    # 创建模型
    print("\n" + "="*60)
    print("构建模型")
    print("="*60)
    model = DashFusion(config).to(device)
    
    # 打印模型摘要
    print_model_summary(model)
    
    # 训练
    print("\n" + "="*60)
    print("开始训练")
    print("="*60)
    trainer = Trainer(model, train_loader, test_loader, config)
    best_test_acc = trainer.train()
    
    # 可视化训练历史
    history_path = os.path.join(config.result_dir, 'history.npy')
    plot_path = os.path.join(config.result_dir, 'training_history.png')
    visualize_training_history(history_path, plot_path)
    
    # 加载最佳模型进行最终测试
    print("\n" + "="*60)
    print("使用最佳模型进行最终测试")
    print("="*60)
    best_model_path = os.path.join(config.ckpt_dir, 'best.pth')
    if os.path.exists(best_model_path):
        model = load_checkpoint(model, best_model_path, device)
        final_test_acc = final_test(model, test_loader, config)
        
        # 保存最终结果
        print("\n" + "="*60)
        print("最终结果")
        print("="*60)
        print(f"训练过程最佳测试准确率: {best_test_acc:.2f}% (Epoch {trainer.best_epoch})")
        print(f"最终测试准确率: {final_test_acc:.2f}%")
        print("="*60 + "\n")
        
        # 保存结果到文件
        with open(os.path.join(config.result_dir, 'final_results.txt'), 'w') as f:
            f.write(f"噪声级别: {config.noise_level}dB\n")
            f.write(f"训练过程最佳测试准确率: {best_test_acc:.2f}% (Epoch {trainer.best_epoch})\n")
            f.write(f"最终测试准确率: {final_test_acc:.2f}%\n")
            f.write(f"模型参数量: {model.get_num_params():,}\n")
            f.write(f"训练样本数: {len(train_loader.dataset)}\n")
            f.write(f"测试样本数: {len(test_loader.dataset)}\n")
    
    print("训练和测试完成！")


if __name__ == '__main__':
    main()