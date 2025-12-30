"""
数据加载器
"""
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os


class RCS_JTF_Dataset(Dataset):
    """
    RCS和JTF数据集
    假设数据格式:
    - rcs: .npy文件, shape: [N, 256]
    - jtf: .npy文件, shape: [N, 256, 256]
    - labels: .npy文件, shape: [N]
    """
    def __init__(self, rcs_path, jtf_path, label_path, transform=None):
        """
        Args:
            rcs_path: RCS数据文件路径 (.npy)
            jtf_path: JTF数据文件路径 (.npy)
            label_path: 标签文件路径 (.npy)
            transform: 数据增强
        """
        self.rcs_data = np.load(rcs_path)
        self.jtf_data = np.load(jtf_path)
        self.labels = np.load(label_path)
        self.transform = transform
        
        assert len(self.rcs_data) == len(self.jtf_data) == len(self.labels), \
            "RCS, JTF and labels must have the same length"
        
        print(f"Loaded dataset: {len(self.rcs_data)} samples")
        print(f"RCS shape: {self.rcs_data.shape}")
        print(f"JTF shape: {self.jtf_data.shape}")
        print(f"Labels shape: {self.labels.shape}")
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        rcs = self.rcs_data[idx]  # [256]
        jtf = self.jtf_data[idx]  # [256, 256]
        label = self.labels[idx]
        
        # 转换为tensor
        rcs = torch.FloatTensor(rcs).unsqueeze(0)  # [1, 256]
        jtf = torch.FloatTensor(jtf).unsqueeze(0)  # [1, 256, 256]
        label = torch.LongTensor([label])[0]
        
        # 数据增强
        if self.transform:
            rcs, jtf = self.transform(rcs, jtf)
        
        return rcs, jtf, label


class RCS_JTF_Transform:
    """数据增强"""
    def __init__(self, noise_level=0.01, time_shift_max=10):
        self.noise_level = noise_level
        self.time_shift_max = time_shift_max
        
    def __call__(self, rcs, jtf):
        # RCS加噪声
        if np.random.rand() < 0.5:
            noise = torch.randn_like(rcs) * self.noise_level
            rcs = rcs + noise
        
        # RCS时移
        if np.random.rand() < 0.5:
            shift = np.random.randint(-self.time_shift_max, self.time_shift_max)
            rcs = torch.roll(rcs, shift, dims=-1)
        
        # JTF加噪声
        if np.random.rand() < 0.5:
            noise = torch.randn_like(jtf) * self.noise_level
            jtf = jtf + noise
        
        return rcs, jtf


def create_dummy_data(save_dir='./dataset/', num_train=1000, num_val=200, num_test=200, num_classes=10):
    """
    创建虚拟数据用于测试
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 训练集
    train_rcs = np.random.randn(num_train, 256).astype(np.float32)
    train_jtf = np.random.randn(num_train, 256, 256).astype(np.float32)
    train_labels = np.random.randint(0, num_classes, num_train)
    
    np.save(os.path.join(save_dir, 'train_rcs.npy'), train_rcs)
    np.save(os.path.join(save_dir, 'train_jtf.npy'), train_jtf)
    np.save(os.path.join(save_dir, 'train_labels.npy'), train_labels)
    
    # 验证集
    val_rcs = np.random.randn(num_val, 256).astype(np.float32)
    val_jtf = np.random.randn(num_val, 256, 256).astype(np.float32)
    val_labels = np.random.randint(0, num_classes, num_val)
    
    np.save(os.path.join(save_dir, 'val_rcs.npy'), val_rcs)
    np.save(os.path.join(save_dir, 'val_jtf.npy'), val_jtf)
    np.save(os.path.join(save_dir, 'val_labels.npy'), val_labels)
    
    # 测试集
    test_rcs = np.random.randn(num_test, 256).astype(np.float32)
    test_jtf = np.random.randn(num_test, 256, 256).astype(np.float32)
    test_labels = np.random.randint(0, num_classes, num_test)
    
    np.save(os.path.join(save_dir, 'test_rcs.npy'), test_rcs)
    np.save(os.path.join(save_dir, 'test_jtf.npy'), test_jtf)
    np.save(os.path.join(save_dir, 'test_labels.npy'), test_labels)
    
    print(f"Dummy data created in {save_dir}")


def get_dataloader(config, split='train'):
    """
    获取数据加载器
    Args:
        config: 配置对象
        split: 'train', 'val', or 'test'
    """
    data_path = config.data_path
    
    # 数据路径
    rcs_path = os.path.join(data_path, f'{split}_rcs.npy')
    jtf_path = os.path.join(data_path, f'{split}_jtf.npy')
    label_path = os.path.join(data_path, f'{split}_labels.npy')
    
    # 检查数据是否存在,不存在则创建虚拟数据
    if not all([os.path.exists(p) for p in [rcs_path, jtf_path, label_path]]):
        print(f"Data not found, creating dummy data...")
        create_dummy_data(data_path, num_classes=config.num_classes)
    
    # 数据增强(仅训练集)
    transform = RCS_JTF_Transform() if split == 'train' else None
    
    # 创建数据集
    dataset = RCS_JTF_Dataset(rcs_path, jtf_path, label_path, transform)
    
    # 创建数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=(split == 'train'),
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=(split == 'train')
    )
    
    return dataloader


if __name__ == '__main__':
    # 测试数据加载器
    class DummyConfig:
        data_path = './dataset/'
        batch_size = 16
        num_workers = 0
        num_classes = 10
    
    config = DummyConfig()
    
    # 创建虚拟数据
    create_dummy_data(config.data_path, num_classes=config.num_classes)
    
    # 测试加载
    train_loader = get_dataloader(config, 'train')
    
    for rcs, jtf, labels in train_loader:
        print(f"RCS shape: {rcs.shape}")
        print(f"JTF shape: {jtf.shape}")
        print(f"Labels shape: {labels.shape}")
        break
