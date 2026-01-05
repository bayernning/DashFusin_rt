"""
训练脚本 - 包含 Hard Negative Mining 逻辑
(使用 utils.py 进行绘图)
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import os
import time
from tqdm import tqdm
import numpy as np

# [修改] 导入 utils 中的可视化函数
from utils import visualize_training_history

class Trainer:
    def __init__(self, model, train_loader, test_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = config
        self.device = config.device

        # ============================================================
        # 1. 优化器配置 (不衰减 Bias 和 Norm 层，提升稳定性)
        # ============================================================
        no_decay = ['bias', 'LayerNorm.weight', 'norm.weight', 'norm1.weight', 'norm2.weight']
        
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': config.weight_decay
            },
            {
                'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]
        
        self.optimizer = optim.AdamW(
            optimizer_grouped_parameters,
            lr=config.learning_rate
        )
        
        # ============================================================
        # 2. 学习率调度器 (Warmup + Cosine)
        # ============================================================
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.01,
            total_iters=config.warmup_steps
        )
        
        total_steps = config.epochs * len(train_loader)
        
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps - config.warmup_steps,
            eta_min=config.learning_rate * 0.01
        )
        
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[config.warmup_steps]
        )
        
        # ============================================================
        # 3. 目录与状态初始化
        # ============================================================
        os.makedirs(config.log_dir, exist_ok=True)
        os.makedirs(config.ckpt_dir, exist_ok=True)
        os.makedirs(config.result_dir, exist_ok=True)
        
        self.best_test_acc = 0.0
        self.best_epoch = 0
        
        # 记录训练曲线
        self.train_losses = []
        self.train_accs = []
        self.test_losses = []
        self.test_accs = []
        self.test_epochs = []

    def update_similarity_matrix(self):
        """
        [核心逻辑] 更新全局相似度矩阵
        遍历整个训练集，提取特征，让 Dataset 重新计算难负样本
        """
        print("\n[Trainer] 正在收集全量特征以更新相似度矩阵 (Hard Negative Mining)...")
        self.model.eval()
        
        # 准备容器: [DatasetSize, Dim]
        # 注意：必须使用 indices 把乱序的 loader 数据填回正确的位置
        dataset_size = len(self.train_loader.dataset)
        hidden_dim = self.config.hidden_dim
        
        all_rcs = torch.zeros(dataset_size, hidden_dim).to(self.device)
        all_tf = torch.zeros(dataset_size, hidden_dim).to(self.device)
        filled_mask = torch.zeros(dataset_size, dtype=torch.bool).to(self.device)
        
        with torch.no_grad():
            # 使用 tqdm 显示进度
            for batch in tqdm(self.train_loader, desc="Updating Matrix"):
                # 解包数据 (兼容 dataset 返回 3个 或 4个 值的情况)
                if len(batch) == 4:
                    rcs, tf, labels, indices = batch
                else:
                    print("错误: Dataset 必须返回 (rcs, tf, labels, indices) 才能使用 Hard Mining!")
                    return

                rcs = rcs.to(self.device)
                tf = tf.to(self.device)
                indices = indices.to(self.device)
                
                # 提取特征 (取均值作为全局表示)
                # 假设 encoder 返回 [B, Seq, Dim]，我们需要 [B, Dim]
                rcs_feat = self.model.rcs_encoder(rcs).mean(dim=1)
                tf_feat = self.model.tf_encoder(tf).mean(dim=1)
                
                all_rcs[indices] = rcs_feat
                all_tf[indices] = tf_feat
                filled_mask[indices] = True
        
        # 检查是否所有数据都覆盖了
        if not filled_mask.all():
            print(f"警告: 有 {(~filled_mask).sum().item()} 个样本未被更新! 请检查 DataLoader 逻辑。")

        # 调用 Dataset 的更新方法 (转回 CPU 以节省显存)
        self.train_loader.dataset.update_matrix(all_rcs.cpu(), all_tf.cpu())
        
        self.model.train()
        print("[Trainer] 相似度矩阵更新完成。\n")

    def train_epoch(self, epoch):
        """训练一个 Epoch"""
        self.model.train()
        total_loss = 0
        total_cls_loss = 0
        total_contrast_loss = 0
        total_correct = 0
        total_samples = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}/{self.config.epochs}')
        
        for batch_idx, batch in enumerate(pbar):
            # 1. 解包数据
            # 注意：Dataset 的 __getitem__ 必须返回 index
            if len(batch) == 4:
                rcs, tf, labels, indices = batch
            else:
                # 兼容旧代码，但这样就没有 Sample2 了
                rcs, tf, labels = batch
                indices = None
            
            rcs = rcs.to(self.device)
            tf = tf.to(self.device)
            labels = labels.to(self.device)
            
            # 2. 获取 Sample2 (难负样本)
            sample2 = None
            if indices is not None:
                # 调用 dataset.sample 获取对应的 6 个辅助样本
                # indices 需要转为 list
                sample2 = self.train_loader.dataset.sample(indices.tolist())
            
            # 3. 前向传播 (传入 sample2)
            outputs = self.model(rcs, tf, labels, sample2=sample2)
            
            loss = outputs['loss']
            cls_loss = outputs.get('cls_loss', torch.tensor(0.0))
            contrast_loss = outputs.get('contrast_loss', torch.tensor(0.0))
            logits = outputs['logits']
            
            # 4. 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            
            self.optimizer.step()
            self.scheduler.step()
            
            # 5. 统计指标
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
        
        # 计算平均值
        avg_loss = total_loss / len(self.train_loader)
        avg_cls_loss = total_cls_loss / len(self.train_loader)
        avg_contrast_loss = total_contrast_loss / len(self.train_loader)
        train_acc = 100.0 * total_correct / total_samples
        
        return avg_loss, avg_cls_loss, avg_contrast_loss, train_acc
    
    @torch.no_grad()
    def test(self):
        """测试模型"""
        self.model.eval()
        
        total_loss = 0
        total_correct = 0
        total_samples = 0
        all_preds = []
        all_labels = []
        
        # 测试集不需要 indices
        for batch in tqdm(self.test_loader, desc='Testing'):
            # 兼容可能返回 index 的情况
            if len(batch) == 4:
                rcs, tf, labels, _ = batch
            else:
                rcs, tf, labels = batch
                
            rcs = rcs.to(self.device)
            tf = tf.to(self.device)
            labels = labels.to(self.device)
            
            # 测试时不需要 sample2
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
        """保存 Checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_test_acc': self.best_test_acc,
            'config': self.config
        }
        
        # 保存 last.pth
        ckpt_path = os.path.join(self.config.ckpt_dir, 'last.pth')
        torch.save(checkpoint, ckpt_path)
        
        # 保存 best.pth
        if is_best:
            best_path = os.path.join(self.config.ckpt_dir, 'best.pth')
            torch.save(checkpoint, best_path)
            print(f'✓ 保存最佳模型，测试准确率: {self.best_test_acc:.2f}%')

    def train(self):
        """主训练流程"""
        print(f"\n{'='*60}")
        print(f"训练 DashFusion on {self.device}")
        print(f"模型参数: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        print(f"{'='*60}\n")
        
        # 为了生成初始的 Sample2，建议在第1轮开始前也更新一次（可选）
        # self.update_similarity_matrix() 
        
        for epoch in range(1, self.config.epochs + 1):
            
            # ========================================================
            # [关键逻辑] 每隔 2 轮更新一次相似度矩阵 (复刻原文)
            # ========================================================
            if (epoch - 1) % 2 == 0:
                self.update_similarity_matrix()
            
            # 训练
            train_loss, cls_loss, con_loss, train_acc = self.train_epoch(epoch)
            
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            
            print(f'\nEpoch {epoch}: Train Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%')
            
            if epoch % self.config.test_interval == 0:
                test_loss, test_acc, preds, labels = self.test()
                self.test_losses.append(test_loss)
                self.test_accs.append(test_acc)
                self.test_epochs.append(epoch)
                
                print(f'  Test Loss: {test_loss:.4f} | Acc: {test_acc:.2f}%')
                
                if test_acc > self.best_test_acc:
                    self.best_test_acc = test_acc
                    self.best_epoch = epoch
                    self.save_checkpoint(epoch, is_best=True)
                
                # 保存预测结果
                np.save(
                    os.path.join(self.config.result_dir, f'predictions_epoch{epoch}.npy'),
                    {'preds': preds, 'labels': labels}
                )
            
            # 定期保存
            if epoch % self.config.save_interval == 0:
                self.save_checkpoint(epoch, is_best=False)
        
        print(f"\n{'='*60}")
        print(f"训练完成! 最佳测试准确率: {self.best_test_acc:.2f}% (Epoch {self.best_epoch})")
        
        # 保存数据
        history_path = os.path.join(self.config.result_dir, 'history.npy')
        history = {
            'train_losses': self.train_losses,
            'train_accs': self.train_accs,
            'test_losses': self.test_losses,
            'test_accs': self.test_accs,
            'test_epochs': self.test_epochs,
            'best_test_acc': self.best_test_acc
        }
        np.save(os.path.join(self.config.result_dir, 'history.npy'), history)
         # [修改] 使用 utils.py 中的函数进行绘图
        try:
            plot_path = os.path.join(self.config.result_dir, 'training_curves.png')
            visualize_training_history(history_path, save_path=plot_path)
        except Exception as e:
            print(f"绘图错误: {e}")
            
        return self.best_test_acc
def final_test(model, test_loader, config):
    # (保持不变)
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
        for batch in tqdm(test_loader, desc='Final Test'):
            # 兼容性解包
            if len(batch) == 4:
                rcs, tf, labels, _ = batch
            else:
                rcs, tf, labels = batch
                
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
        'accuracy': test_acc
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