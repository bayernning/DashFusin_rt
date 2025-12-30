"""
训练脚本
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
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
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
        
        # 最佳验证准确率
        self.best_val_acc = 0.0
        self.best_epoch = 0
        
        # 日志
        self.train_losses = []
        self.val_losses = []
        self.val_accs = []
        
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        
        total_loss = 0
        total_cls_loss = 0
        total_contrast_loss = 0
        total_correct = 0
        total_samples = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}/{self.config.epochs}')
        
        for batch_idx, (rcs, jtf, labels) in enumerate(pbar):
            # 数据移到设备
            rcs = rcs.to(self.device)
            jtf = jtf.to(self.device)
            labels = labels.to(self.device)
            
            # 前向传播
            outputs = self.model(rcs, jtf, labels)
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
                'cls_loss': f'{cls_loss.item():.4f}',
                'con_loss': f'{contrast_loss.item():.4f}',
                'acc': f'{100.0 * total_correct / total_samples:.2f}%',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })
        
        avg_loss = total_loss / len(self.train_loader)
        avg_cls_loss = total_cls_loss / len(self.train_loader)
        avg_contrast_loss = total_contrast_loss / len(self.train_loader)
        train_acc = 100.0 * total_correct / total_samples
        
        return avg_loss, avg_cls_loss, avg_contrast_loss, train_acc
    
    @torch.no_grad()
    def validate(self):
        """验证"""
        self.model.eval()
        
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        for rcs, jtf, labels in tqdm(self.val_loader, desc='Validation'):
            rcs = rcs.to(self.device)
            jtf = jtf.to(self.device)
            labels = labels.to(self.device)
            
            outputs = self.model(rcs, jtf, labels)
            loss = outputs['loss']
            logits = outputs['logits']
            
            total_loss += loss.item()
            
            preds = logits.argmax(dim=-1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
        
        avg_loss = total_loss / len(self.val_loader)
        val_acc = 100.0 * total_correct / total_samples
        
        return avg_loss, val_acc
    
    def save_checkpoint(self, epoch, is_best=False):
        """保存checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_acc': self.best_val_acc,
            'config': self.config
        }
        
        # 保存最新checkpoint
        ckpt_path = os.path.join(self.config.ckpt_dir, 'last.pth')
        torch.save(checkpoint, ckpt_path)
        
        # 保存最佳checkpoint
        if is_best:
            best_path = os.path.join(self.config.ckpt_dir, 'best.pth')
            torch.save(checkpoint, best_path)
            print(f'✓ Saved best checkpoint with val_acc: {self.best_val_acc:.2f}%')
    
    def train(self):
        """完整训练流程"""
        print(f"\n{'='*60}")
        print(f"Training DashFusion on {self.device}")
        print(f"Model parameters: {self.model.get_num_params():,}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        print(f"{'='*60}\n")
        
        for epoch in range(1, self.config.epochs + 1):
            # 训练
            train_loss, cls_loss, con_loss, train_acc = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            
            # 验证
            val_loss, val_acc = self.validate()
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)
            
            # 打印结果
            print(f'\nEpoch {epoch}/{self.config.epochs}:')
            print(f'  Train Loss: {train_loss:.4f} (cls: {cls_loss:.4f}, con: {con_loss:.4f})')
            print(f'  Train Acc:  {train_acc:.2f}%')
            print(f'  Val Loss:   {val_loss:.4f}')
            print(f'  Val Acc:    {val_acc:.2f}%')
            
            # 保存最佳模型
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
            
            # 定期保存
            if epoch % self.config.save_interval == 0 or is_best:
                self.save_checkpoint(epoch, is_best)
        
        print(f"\n{'='*60}")
        print(f"Training completed!")
        print(f"Best validation accuracy: {self.best_val_acc:.2f}% (Epoch {self.best_epoch})")
        print(f"{'='*60}\n")
        
        # 保存训练历史
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accs': self.val_accs,
            'best_val_acc': self.best_val_acc,
            'best_epoch': self.best_epoch
        }
        np.save(os.path.join(self.config.result_dir, 'history.npy'), history)
        
        return self.best_val_acc


def test(model, test_loader, config):
    """测试函数"""
    model.eval()
    device = config.device
    
    total_correct = 0
    total_samples = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for rcs, jtf, labels in tqdm(test_loader, desc='Testing'):
            rcs = rcs.to(device)
            jtf = jtf.to(device)
            labels = labels.to(device)
            
            outputs = model(rcs, jtf)
            logits = outputs['logits']
            preds = logits.argmax(dim=-1)
            
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    test_acc = 100.0 * total_correct / total_samples
    
    print(f"\n{'='*60}")
    print(f"Test Results:")
    print(f"  Test Accuracy: {test_acc:.2f}%")
    print(f"  Total Samples: {total_samples}")
    print(f"{'='*60}\n")
    
    # 保存预测结果
    results = {
        'predictions': all_preds,
        'labels': all_labels,
        'accuracy': test_acc
    }
    np.save(os.path.join(config.result_dir, 'test_results.npy'), results)
    
    return test_acc


def load_checkpoint(model, ckpt_path, device):
    """加载checkpoint"""
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from {ckpt_path}")
    print(f"Epoch: {checkpoint['epoch']}, Best Val Acc: {checkpoint['best_val_acc']:.2f}%")
    return model
