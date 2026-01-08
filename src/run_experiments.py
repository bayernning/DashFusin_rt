"""
自动化批量实验脚本
运行多组实验配置，保存结果并生成对比报告
"""
import torch
import os
import sys
import json
import time
import pandas as pd
from datetime import datetime

from config import get_config
from model.dashfusion import DashFusion
# from dataloader.dataset_12 import get_dataloader, check_data_format
from dataloader.dataset_modified_v2 import get_dataloader, check_data_format
from train import Trainer, final_test, load_checkpoint
from utils import set_seed, save_config
from experiment_configs import EXPERIMENT_CONFIGS, list_experiments


class ExperimentRunner:
    """实验运行器"""
    
    def __init__(self, base_config, save_dir='./experiments'):
        self.base_config = base_config
        self.save_dir = save_dir
        self.results = []
        
        # 创建实验根目录
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 创建时间戳
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.session_dir = os.path.join(self.save_dir, f'session_{self.timestamp}')
        os.makedirs(self.session_dir, exist_ok=True)
        
        print(f"\n{'='*80}")
        print(f"实验会话目录: {self.session_dir}")
        print(f"{'='*80}\n")
    
    def run_single_experiment(self, exp_config):
        """运行单个实验"""
        exp_name = exp_config['name']
        exp_params = exp_config['params']
        
        print(f"\n{'='*80}")
        print(f"开始实验: {exp_name}")
        print(f"描述: {exp_config['description']}")
        print(f"{'='*80}\n")
        
        # 创建实验目录
        exp_dir = os.path.join(self.session_dir, exp_name)
        os.makedirs(exp_dir, exist_ok=True)
        
        # 更新配置
        config = self.base_config
        for key, value in exp_params.items():
            setattr(config, key, value)
        
        # 更新保存路径
        config.log_dir = os.path.join(exp_dir, 'log')
        config.ckpt_dir = os.path.join(exp_dir, 'ckpt')
        config.result_dir = os.path.join(exp_dir, 'result')
        
        os.makedirs(config.log_dir, exist_ok=True)
        os.makedirs(config.ckpt_dir, exist_ok=True)
        os.makedirs(config.result_dir, exist_ok=True)
        
        # 保存配置
        save_config(config, os.path.join(exp_dir, 'config.json'))
        
        # 保存实验描述
        with open(os.path.join(exp_dir, 'description.txt'), 'w') as f:
            f.write(f"实验名称: {exp_name}\n")
            f.write(f"描述: {exp_config['description']}\n\n")
            f.write("参数配置:\n")
            for key, value in exp_params.items():
                f.write(f"  {key}: {value}\n")
        
        try:
            # 设置随机种子
            set_seed(config.seed)
            
            # 设置设备
            device = torch.device(config.device)
            
            # 创建数据加载器
            train_loader = get_dataloader(config, split='train')
            test_loader = get_dataloader(config, split='test')
            
            # 创建模型
            model = DashFusion(config).to(device)
            
            # 训练
            start_time = time.time()
            trainer = Trainer(model, train_loader, test_loader, config)
            best_test_acc = trainer.train()
            training_time = time.time() - start_time
            
            # 最终测试
            best_model_path = os.path.join(config.ckpt_dir, 'best.pth')
            if os.path.exists(best_model_path):
                model = load_checkpoint(model, best_model_path, device)
                final_test_acc = final_test(model, test_loader, config)
            else:
                final_test_acc = best_test_acc
            
            # 记录结果
            result = {
                'experiment': exp_name,
                'description': exp_config['description'],
                'best_epoch': trainer.best_epoch,
                'best_train_acc': max(trainer.train_accs),
                'best_test_acc': best_test_acc,
                'final_test_acc': final_test_acc,
                'training_time_minutes': training_time / 60,
                'num_params': model.get_num_params(),
                **exp_params  # 包含所有超参数
            }
            
            self.results.append(result)
            
            # 保存单个实验结果
            with open(os.path.join(exp_dir, 'result.json'), 'w') as f:
                json.dump(result, f, indent=4)
            
            print(f"\n{'='*80}")
            print(f"实验 {exp_name} 完成!")
            print(f"最佳测试准确率: {best_test_acc:.2f}% (Epoch {trainer.best_epoch})")
            print(f"最终测试准确率: {final_test_acc:.2f}%")
            print(f"训练时间: {training_time/60:.1f} 分钟")
            print(f"{'='*80}\n")
            
            # 清理显存
            del model, trainer, train_loader, test_loader
            torch.cuda.empty_cache()
            
            return True, result
            
        except Exception as e:
            print(f"\n{'='*80}")
            print(f"实验 {exp_name} 失败!")
            print(f"错误: {e}")
            print(f"{'='*80}\n")
            
            import traceback
            traceback.print_exc()
            
            # 记录失败
            result = {
                'experiment': exp_name,
                'status': 'failed',
                'error': str(e),
                **exp_params
            }
            self.results.append(result)
            
            return False, result
    
    def run_all_experiments(self, experiment_names=None):
        """运行所有实验"""
        # 选择要运行的实验
        if experiment_names is None:
            experiments = EXPERIMENT_CONFIGS
        else:
            experiments = [cfg for cfg in EXPERIMENT_CONFIGS if cfg['name'] in experiment_names]
        
        print(f"\n{'='*80}")
        print(f"批量实验开始")
        print(f"总共 {len(experiments)} 个实验")
        print(f"{'='*80}\n")
        
        total_start = time.time()
        
        for i, exp_config in enumerate(experiments, 1):
            print(f"\n>>> 进度: {i}/{len(experiments)} <<<\n")
            self.run_single_experiment(exp_config)
        
        total_time = time.time() - total_start
        
        # 生成对比报告
        self.generate_comparison_report()
        
        print(f"\n{'='*80}")
        print(f"所有实验完成!")
        print(f"总耗时: {total_time/60:.1f} 分钟")
        print(f"结果保存在: {self.session_dir}")
        print(f"{'='*80}\n")
    
    def generate_comparison_report(self):
        """生成对比报告"""
        if not self.results:
            print("没有实验结果可以生成报告")
            return
        
        # 创建DataFrame
        df = pd.DataFrame(self.results)
        
        # 保存CSV
        csv_path = os.path.join(self.session_dir, 'comparison_results.csv')
        df.to_csv(csv_path, index=False)
        print(f"✓ 对比结果CSV保存到: {csv_path}")
        
        # 保存JSON
        json_path = os.path.join(self.session_dir, 'comparison_results.json')
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=4)
        print(f"✓ 对比结果JSON保存到: {json_path}")
        
        # 生成Markdown报告
        self.generate_markdown_report(df)
        
        # 打印总结
        self.print_summary(df)
    
    def generate_markdown_report(self, df):
        """生成Markdown报告"""
        md_path = os.path.join(self.session_dir, 'REPORT.md')
        
        with open(md_path, 'w') as f:
            f.write(f"# DashFusion 实验对比报告\n\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # 总体统计
            f.write("## 总体统计\n\n")
            f.write(f"- 实验数量: {len(df)}\n")
            f.write(f"- 最高测试准确率: {df['final_test_acc'].max():.2f}%\n")
            f.write(f"- 最佳实验: {df.loc[df['final_test_acc'].idxmax(), 'experiment']}\n")
            f.write(f"- 平均训练时间: {df['training_time_minutes'].mean():.1f} 分钟\n\n")
            
            # 排名表
            f.write("## 实验结果排名（按最终测试准确率）\n\n")
            df_sorted = df.sort_values('final_test_acc', ascending=False)
            
            f.write("| 排名 | 实验名称 | 测试准确率 | 训练准确率 | 最佳Epoch | 训练时间 |\n")
            f.write("|------|----------|-----------|-----------|----------|----------|\n")
            
            for idx, row in df_sorted.iterrows():
                f.write(f"| {df_sorted.index.get_loc(idx)+1} | {row['experiment']} | "
                       f"{row['final_test_acc']:.2f}% | {row['best_train_acc']:.2f}% | "
                       f"{row['best_epoch']} | {row['training_time_minutes']:.1f}min |\n")
            
            # 详细对比
            f.write("\n## 详细配置对比\n\n")
            f.write("| 实验 | 学习率 | Batch | 对比权重 | 温度 | 瓶颈数 | 融合层数 | Dropout |\n")
            f.write("|------|--------|-------|---------|------|--------|---------|--------|\n")
            
            for _, row in df_sorted.iterrows():
                f.write(f"| {row['experiment']} | {row['learning_rate']:.0e} | "
                       f"{row['batch_size']} | {row['contrast_loss_weight']:.2f} | "
                       f"{row['temperature']:.2f} | {row['num_bottleneck']} | "
                       f"{row['num_fusion_layers']} | {row['dropout']:.2f} |\n")
            
            # 各实验详细描述
            f.write("\n## 各实验详细说明\n\n")
            for _, row in df.iterrows():
                f.write(f"### {row['experiment']}\n\n")
                f.write(f"**描述**: {row['description']}\n\n")
                f.write(f"**结果**:\n")
                f.write(f"- 最终测试准确率: {row['final_test_acc']:.2f}%\n")
                f.write(f"- 最佳训练准确率: {row['best_train_acc']:.2f}%\n")
                f.write(f"- 最佳Epoch: {row['best_epoch']}\n")
                f.write(f"- 训练时间: {row['training_time_minutes']:.1f} 分钟\n")
                f.write(f"- 模型参数量: {row['num_params']:,}\n\n")
            
            # 建议
            f.write("\n## 结论与建议\n\n")
            best_exp = df.loc[df['final_test_acc'].idxmax()]
            f.write(f"1. **最佳配置**: {best_exp['experiment']}\n")
            f.write(f"   - 测试准确率: {best_exp['final_test_acc']:.2f}%\n")
            f.write(f"   - 学习率: {best_exp['learning_rate']}\n")
            f.write(f"   - 批次大小: {best_exp['batch_size']}\n")
            f.write(f"   - 对比学习权重: {best_exp['contrast_loss_weight']}\n\n")
            
            f.write("2. **参数影响分析**:\n")
            f.write("   - 查看CSV文件进行更详细的参数相关性分析\n")
            f.write("   - 可以基于最佳配置进行微调\n\n")
        
        print(f"✓ Markdown报告保存到: {md_path}")
    
    def print_summary(self, df):
        """打印总结"""
        print("\n" + "="*80)
        print("实验总结")
        print("="*80)
        
        df_sorted = df.sort_values('final_test_acc', ascending=False)
        
        print("\n📊 TOP 3 配置:\n")
        for i, (_, row) in enumerate(df_sorted.head(3).iterrows(), 1):
            print(f"{i}. {row['experiment']}")
            print(f"   测试准确率: {row['final_test_acc']:.2f}%")
            print(f"   关键参数: lr={row['learning_rate']}, "
                  f"batch={row['batch_size']}, "
                  f"contrast_weight={row['contrast_loss_weight']}")
            print()
        
        print("="*80 + "\n")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='批量运行DashFusion实验')
    parser.add_argument('--list', action='store_true', help='列出所有可用的实验配置')
    parser.add_argument('--experiments', nargs='+', help='指定要运行的实验名称（留空则运行所有）')
    parser.add_argument('--save_dir', type=str, default='./experiments', help='实验结果保存目录')
    
    args = parser.parse_args()
    
    # 列出实验
    if args.list:
        list_experiments()
        return
    
    # 获取基础配置
    base_config = get_config()
    
    # 检查数据
    print("\n" + "="*80)
    print("检查数据格式")
    print("="*80)
    if not check_data_format(base_config):
        print("数据格式检查失败，请检查数据文件！")
        return
    
    # 创建实验运行器
    runner = ExperimentRunner(base_config, args.save_dir)
    
    # 运行实验
    runner.run_all_experiments(args.experiments)


if __name__ == '__main__':
    main()
