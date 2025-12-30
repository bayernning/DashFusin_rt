"""
训练脚本 - 每10轮测试一次
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import os
import time
from tqdm import tqdm
import numpy as np


class Trainer:
    def __init__(self, model, train_loader, test_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = config
        self.device = config.device
        
        # 优化器
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # 学习率调度器: warmup + cosine annealing
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            total_iters=config.warmup_steps
        )
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.epochs * len(train_loader) - config.warmup_steps
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[config.warmup_steps]
        )
        
        # 创建保存目录
        os.makedirs(config.log_dir, exist_ok=True)
        os.makedirs(config.ckpt_dir, exist_ok=True)
        os.makedirs(config.result_dir, exist_ok=True)
        
        # 最佳测试准确率
        self.best_test_acc = 0.0
        self.best_epoch = 0
        
        # 日志
        self.train_losses = []
        self.train_accs = []
        self.test_losses = []
        self.test_accs = []
        self.test_epochs = []  # 记录测试的epoch
        
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        
        total_loss = 0
        total_cls_loss = 0
        total_contrast_loss = 0
        total_correct = 0
        total_samples = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}/{self.config.epochs}')
        
        for batch_idx, (rcs, tf, labels) in enumerate(pbar):
            # 数据移到设备
            rcs = rcs.to(self.device)
            tf = tf.to(self.device)
            labels = labels.to(self.device)
            
            # 前向传播
            outputs = self.model(rcs, tf, labels)
            loss = outputs['loss']
            cls_loss = outputs['cls_loss']
            contrast_loss = outputs['contrast_loss']
            logits = outputs['logits']
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            
            self.optimizer.step()
            self.scheduler.step()
            
            # 统计
            total_loss += loss.item()
            total_cls_loss += cls_loss.item()
            total_contrast_loss += contrast_loss.item()
            
            preds = logits.argmax(dim=-1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'cls': f'{cls_loss.item():.4f}',
                'con': f'{contrast_loss.item():.4f}',
                'acc': f'{100.0 * total_correct / total_samples:.2f}%',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })
        
        avg_loss = total_loss / len(self.train_loader)
        avg_cls_loss = total_cls_loss / len(self.train_loader)
        avg_contrast_loss = total_contrast_loss / len(self.train_loader)
        train_acc = 100.0 * total_correct / total_samples
        
        return avg_loss, avg_cls_loss, avg_contrast_loss, train_acc
    
    @torch.no_grad()
    def test(self):
        """测试"""
        self.model.eval()
        
        total_loss = 0
        total_correct = 0
        total_samples = 0
        all_preds = []
        all_labels = []
        
        for rcs, tf, labels in tqdm(self.test_loader, desc='Testing'):
            rcs = rcs.to(self.device)
            tf = tf.to(self.device)
            labels = labels.to(self.device)
            
            outputs = self.model(rcs, tf, labels)
            loss = outputs['loss']
            logits = outputs['logits']
            
            total_loss += loss.item()
            
            preds = logits.argmax(dim=-1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(self.test_loader)
        test_acc = 100.0 * total_correct / total_samples
        
        return avg_loss, test_acc, all_preds, all_labels
    
    def save_checkpoint(self, epoch, is_best=False):
        """保存checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_test_acc': self.best_test_acc,
            'config': self.config
        }
        
        # 保存最新checkpoint
        ckpt_path = os.path.join(self.config.ckpt_dir, 'last.pth')
        torch.save(checkpoint, ckpt_path)
        
        # 保存最佳checkpoint
        if is_best:
            best_path = os.path.join(self.config.ckpt_dir, 'best.pth')
            torch.save(checkpoint, best_path)
            print(f'✓ 保存最佳模型，测试准确率: {self.best_test_acc:.2f}%')
    
    def train(self):
        """完整训练流程"""
        print(f"\n{'='*60}")
        print(f"训练DashFusion on {self.device}")
        print(f"模型参数: {self.model.get_num_params():,}")
        print(f"训练样本: {len(self.train_loader.dataset)}")
        print(f"测试样本: {len(self.test_loader.dataset)}")
        print(f"测试间隔: 每{self.config.test_interval}轮")
        print(f"{'='*60}\n")
        
        for epoch in range(1, self.config.epochs + 1):
            # 训练
            train_loss, cls_loss, con_loss, train_acc = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            
            # 打印训练结果
            print(f'\nEpoch {epoch}/{self.config.epochs}:')
            print(f'  Train Loss: {train_loss:.4f} (cls: {cls_loss:.4f}, con: {con_loss:.4f})')
            print(f'  Train Acc:  {train_acc:.2f}%')
            
            # 每10轮测试一次
            if epoch % self.config.test_interval == 0:
                test_loss, test_acc, preds, labels = self.test()
                self.test_losses.append(test_loss)
                self.test_accs.append(test_acc)
                self.test_epochs.append(epoch)
                
                print(f'  Test Loss:  {test_loss:.4f}')
                print(f'  Test Acc:   {test_acc:.2f}%')
                
                # 保存最佳模型
                is_best = test_acc > self.best_test_acc
                if is_best:
                    self.best_test_acc = test_acc
                    self.best_epoch = epoch
                    self.save_checkpoint(epoch, is_best=True)
                
                # 保存预测结果
                np.save(
                    os.path.join(self.config.result_dir, f'predictions_epoch{epoch}.npy'),
                    {'preds': preds, 'labels': labels}
                )
            
            # 定期保存checkpoint
            if epoch % self.config.save_interval == 0:
                self.save_checkpoint(epoch, is_best=False)
        
        print(f"\n{'='*60}")
        print(f"训练完成!")
        print(f"最佳测试准确率: {self.best_test_acc:.2f}% (Epoch {self.best_epoch})")
        print(f"{'='*60}\n")
        
        # 保存训练历史
        history = {
            'train_losses': self.train_losses,
            'train_accs': self.train_accs,
            'test_losses': self.test_losses,
            'test_accs': self.test_accs,
            'test_epochs': self.test_epochs,
            'best_test_acc': self.best_test_acc,
            'best_epoch': self.best_epoch
        }
        np.save(os.path.join(self.config.result_dir, 'history.npy'), history)
        
        return self.best_test_acc


def final_test(model, test_loader, config):
    """最终测试函数"""
    model.eval()
    device = config.device
    
    total_correct = 0
    total_samples = 0
    all_preds = []
    all_labels = []
    
    # 类别统计
    num_classes = config.num_classes
    class_correct = [0] * num_classes
    class_total = [0] * num_classes
    
    with torch.no_grad():
        for rcs, tf, labels in tqdm(test_loader, desc='最终测试'):
            rcs = rcs.to(device)
            tf = tf.to(device)
            labels = labels.to(device)
            
            outputs = model(rcs, tf)
            logits = outputs['logits']
            preds = logits.argmax(dim=-1)
            
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # 统计每个类别
            for i in range(len(labels)):
                label = labels[i].item()
                class_total[label] += 1
                if preds[i] == labels[i]:
                    class_correct[label] += 1
    
    test_acc = 100.0 * total_correct / total_samples
    
    print(f"\n{'='*60}")
    print(f"最终测试结果:")
    print(f"  总体准确率: {test_acc:.2f}%")
    print(f"  总样本数: {total_samples}")
    print(f"\n各类别准确率:")
    for i in range(num_classes):
        if class_total[i] > 0:
            acc = 100.0 * class_correct[i] / class_total[i]
            print(f"  类别 {i}: {acc:.2f}% ({class_correct[i]}/{class_total[i]})")
    print(f"{'='*60}\n")
    
    # 保存预测结果
    results = {
        'predictions': all_preds,
        'labels': all_labels,
        'accuracy': test_acc,
        'class_correct': class_correct,
        'class_total': class_total
    }
    np.save(os.path.join(config.result_dir, 'final_test_results.npy'), results)
    
    return test_acc


def load_checkpoint(model, ckpt_path, device):
    """加载checkpoint"""
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"加载checkpoint: {ckpt_path}")
    print(f"Epoch: {checkpoint['epoch']}, 最佳测试准确率: {checkpoint['best_test_acc']:.2f}%")
    return model