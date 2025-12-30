"""
数据加载器 - 适配RCS和TF联合数据集
"""
import os
import torch
from torch.utils.data import Dataset, DataLoader
import scipy.io as sio
import numpy as np


class RCSTFJointDataset(Dataset):
    """
    RCS和TF联合训练数据集
    """
    def __init__(self, rcs_mat_path, tf_images, tf_labels, tf_transform=None):
        """
        Args:
            rcs_mat_path: RCS的.mat文件路径
            tf_images: TF图像tensor [N, 256, 256]
            tf_labels: TF标签tensor [N]
            tf_transform: TF图像的变换
        """
        # -------------------------
        # 1) 读取 RCS 数据
        # -------------------------
        mat_data = sio.loadmat(rcs_mat_path)
        data_train = mat_data["data_train"]
        
        self.num_samples = data_train.shape[1]
        self.rcs_data = []
        self.rcs_labels = []

        for i in range(self.num_samples):
            rcs = data_train[0, i]["im_ori"].squeeze()     # (256,)
            label = data_train[0, i]["idx"].squeeze()

            # 归一化
            rcs = np.abs(rcs) / np.max(np.abs(rcs))

            self.rcs_data.append(rcs)
            self.rcs_labels.append(label)

        self.rcs_data = torch.tensor(np.stack(self.rcs_data), dtype=torch.float32)   # (N,256)
        self.rcs_labels = torch.tensor(np.array(self.rcs_labels), dtype=torch.long)  # (N,)

        # -------------------------
        # 2) TF 数据
        # -------------------------
        self.tf_images = tf_images.float().unsqueeze(1)  # (N,1,H,W)
        self.tf_labels = tf_labels.long()
        self.tf_transform = tf_transform

        assert len(self.rcs_data) == len(self.tf_images), \
            f"RCS样本数({len(self.rcs_data)})必须等于TF样本数({len(self.tf_images)})"
        
        print(f"[训练集] 加载完成:")
        print(f"  - RCS数据: {self.rcs_data.shape}")
        print(f"  - TF图像: {self.tf_images.shape}")
        print(f"  - 标签: {self.rcs_labels.shape}")
        print(f"  - 样本总数: {len(self.rcs_data)}")

    def __len__(self):
        return len(self.rcs_data)

    def __getitem__(self, idx):
        # ---------- RCS ----------
        rcs = self.rcs_data[idx]         # (256,)
        rcs = rcs.unsqueeze(0)           # → (1,256)
        label = int(self.rcs_labels[idx])

        # ---------- TF ----------
        tf_img = self.tf_images[idx]     # (1,H,W)
        if self.tf_transform:
            tf_img = self.tf_transform(tf_img)

        return rcs, tf_img, label


class Test_RCSTFJointDataset(Dataset):
    """
    RCS和TF联合测试数据集
    """
    def __init__(self, rcs_mat_path, tf_images, tf_labels, tf_transform=None):
        """
        Args:
            rcs_mat_path: RCS的.mat文件路径
            tf_images: TF图像tensor [N, 256, 256]
            tf_labels: TF标签tensor [N]
            tf_transform: TF图像的变换
        """
        # -------------------------
        # 1) 读取 RCS 数据
        # -------------------------
        mat_data = sio.loadmat(rcs_mat_path)
        data_test = mat_data["data_test"]
        
        self.num_samples = data_test.shape[1]
        self.rcs_data = []
        self.rcs_labels = []

        for i in range(self.num_samples):
            rcs = data_test[0, i]["im_ori"].squeeze()     # (256,)
            label = data_test[0, i]["idx"].squeeze()

            # 归一化
            rcs = np.abs(rcs) / np.max(np.abs(rcs))

            self.rcs_data.append(rcs)
            self.rcs_labels.append(label)

        self.rcs_data = torch.tensor(np.stack(self.rcs_data), dtype=torch.float32)   # (N,256)
        self.rcs_labels = torch.tensor(np.array(self.rcs_labels), dtype=torch.long)  # (N,)

        # -------------------------
        # 2) TF 数据
        # -------------------------
        self.tf_images = tf_images.float().unsqueeze(1)  # (N,1,H,W)
        self.tf_labels = tf_labels.long()
        self.tf_transform = tf_transform

        assert len(self.rcs_data) == len(self.tf_images), \
            f"RCS样本数({len(self.rcs_data)})必须等于TF样本数({len(self.tf_images)})"
        
        print(f"[测试集] 加载完成:")
        print(f"  - RCS数据: {self.rcs_data.shape}")
        print(f"  - TF图像: {self.tf_images.shape}")
        print(f"  - 标签: {self.rcs_labels.shape}")
        print(f"  - 样本总数: {len(self.rcs_data)}")

    def __len__(self):
        return len(self.rcs_data)

    def __getitem__(self, idx):
        # ---------- RCS ----------
        rcs = self.rcs_data[idx]         # (256,)
        rcs = rcs.unsqueeze(0)           # → (1,256)
        label = int(self.rcs_labels[idx])

        # ---------- TF ----------
        tf_img = self.tf_images[idx]     # (1,H,W)
        if self.tf_transform:
            tf_img = self.tf_transform(tf_img)

        return rcs, tf_img, label


def get_dataloader(config, split='train'):
    """
    获取数据加载器
    Args:
        config: 配置对象
        split: 'train' or 'test'
    """
    noise = config.noise_level
    
    # 构建数据路径
    if split == 'train':
        pt_path = os.path.join(config.train_data_dir, f"train_patch_{noise}dB.pt")
        mat_path = os.path.join(config.train_data_dir, f"rcs_data_300_{noise}dB.mat")
    else:  # test
        pt_path = os.path.join(config.test_data_dir, f"test_patch_{noise}dB.pt")
        mat_path = os.path.join(config.test_data_dir, f"rcs_data1_300_{noise}dB.mat")
    
    # 检查文件是否存在
    if not os.path.exists(pt_path):
        raise FileNotFoundError(f"找不到数据文件: {pt_path}")
    if not os.path.exists(mat_path):
        raise FileNotFoundError(f"找不到数据文件: {mat_path}")
    
    print(f"\n{'='*60}")
    print(f"加载{split}数据:")
    print(f"  - PT文件: {pt_path}")
    print(f"  - MAT文件: {mat_path}")
    print(f"{'='*60}")
    
    # 加载 .pt 数据
    data = torch.load(pt_path)
    tf_images = data["image"]
    tf_labels = data["label"]
    
    # 创建数据集
    if split == 'train':
        dataset = RCSTFJointDataset(
            rcs_mat_path=mat_path,
            tf_images=tf_images,
            tf_labels=tf_labels,
            tf_transform=None  # 可以在这里添加数据增强
        )
    else:  # test
        dataset = Test_RCSTFJointDataset(
            rcs_mat_path=mat_path,
            tf_images=tf_images,
            tf_labels=tf_labels,
            tf_transform=None
        )
    
    # 创建数据加载器
    shuffle = (split == 'train')
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=(split == 'train')
    )
    
    return dataloader


# ============= 辅助函数 =============

def check_data_format(config):
    """
    检查数据格式是否正确
    """
    print("\n" + "="*60)
    print("检查数据格式")
    print("="*60)
    
    try:
        train_loader = get_dataloader(config, 'train')
        test_loader = get_dataloader(config, 'test')
        
        # 测试一个batch
        rcs, tf, labels = next(iter(train_loader))
        
        print(f"\n✓ 数据格式检查通过:")
        print(f"  - RCS shape: {rcs.shape} (期望: [batch, 1, 256])")
        print(f"  - TF shape: {tf.shape} (期望: [batch, 1, H, W])")
        print(f"  - Labels shape: {labels.shape} (期望: [batch])")
        print(f"  - RCS范围: [{rcs.min():.3f}, {rcs.max():.3f}]")
        print(f"  - TF范围: [{tf.min():.3f}, {tf.max():.3f}]")
        print(f"  - 标签唯一值: {torch.unique(labels).tolist()}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ 数据格式检查失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    # 测试数据加载器
    print("测试数据加载器...")
    
    class DummyConfig:
        noise_level = 0
        train_data_dir = "./train_data"
        test_data_dir = "./test_data"
        batch_size = 16
        num_workers = 2
    
    config = DummyConfig()
    
    # 检查数据格式
    if check_data_format(config):
        print("\n✓ 数据加载器测试成功！")
    else:
        print("\n✗ 数据加载器测试失败！")