"""
数据加载器 - 适配RCS和TF联合数据集 (包含 Hard Negative Mining 支持)
"""
import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import scipy.io as sio
import numpy as np
import random
from torchvision import transforms

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
            
        # -------------------------
        # 3) 检索相关初始化 (Hard Mining)
        # -------------------------
        self.size = len(self.rcs_data)
        self.M_retrieve = None # 存储检索结果
        
        print(f"[训练集] 加载完成:")
        print(f"  - RCS数据: {self.rcs_data.shape}")
        print(f"  - TF图像: {self.tf_images.shape}")
        print(f"  - 标签: {self.rcs_labels.shape}")
        print(f"  - 样本总数: {len(self.rcs_data)}")

    # [新增] 更新相似度矩阵 (由 Trainer 调用)
    def update_matrix(self, rcs_emb, tf_emb):
        """
        rcs_emb: [N, Dim] 全局RCS特征
        tf_emb: [N, Dim] 全局TF特征
        """
        # 1. 特征归一化并拼接
        rcs_emb = F.normalize(rcs_emb, p=2, dim=1)
        tf_emb = F.normalize(tf_emb, p=2, dim=1)
        # 融合特征用于计算相似度
        joint_emb = torch.cat((rcs_emb, tf_emb), dim=1) 
        
        # 2. 计算全局余弦相似度矩阵 [N, N]
        # 注意: 这里尽量在CPU上做以防显存溢出
        joint_emb = joint_emb.cpu()
        cos_matrix = torch.matmul(joint_emb, joint_emb.T)
        
        # 3. 排序 (Descending)
        _, rank_M = torch.sort(cos_matrix, descending=True, dim=1)
        
        # 4. 执行预采样分类
        self.__pre_sample(rank_M, self.rcs_labels.cpu())

    # [新增] 预采样逻辑 (Hard Mining 核心)
    def __pre_sample(self, _rank, _label):
        retrieve = {'ss': [], 'sd': [], 'ds': [], 'dd': []}
        
        for i in range(self.size):
            _ss, _sd, _dd = [], [], []
            
            # 策略: 在前 60% 相似的样本中找 Positive 和 Hard Negative
            search_range = int(self.size / 1.6)
            for j in range(search_range):
                idx = int(_rank[i][j])
                if i == idx: continue # 跳过自己
                
                if _label[i] == _label[idx]:
                    _ss.append(idx) # Similar & Same (强正样本)
                else:
                    _sd.append(idx) # Similar & Diff (难负样本)
            
            # 策略: 在最不相似的尾部找 Easy Negative
            for j in range(self.size - 1, self.size - int(self.size/2), -1):
                idx = int(_rank[i][j])
                if i == idx: continue
                if _label[i] != _label[idx]:
                    _dd.append(idx) # Dissimilar & Diff (易负样本)
            
            # 兜底填充 (防止某种类型样本不足)
            while len(_ss) < 2: _ss.append(random.randint(0, self.size-1))
            while len(_sd) < 2: _sd.append(random.randint(0, self.size-1))
            while len(_dd) < 2: _dd.append(random.randint(0, self.size-1))

            retrieve['ss'].append(_ss)
            retrieve['sd'].append(_sd)
            retrieve['dd'].append(_dd)
            
        self.M_retrieve = retrieve
        print(f"[Dataset] Hard Negatives Mined. Ready for sampling.")

    # [新增] 采样函数
    def sample(self, indices):
        """
        根据索引列表返回对应的 sample2
        indices: list of int
        """
        if self.M_retrieve is None:
            return None # 还没初始化矩阵

        # 收集所有索引
        batch_indices = []
        for i in indices:
            i = int(i)
            # 随机抽 2个SS, 2个SD, 2个DD
            ss = random.choices(self.M_retrieve['ss'][i], k=1)
            sd = random.choices(self.M_retrieve['sd'][i], k=1)
            dd = random.choices(self.M_retrieve['dd'][i], k=1)
            batch_indices.extend(ss + sd + dd)
            
        # 批量取数据
        samples2 = {
            'rcs': self.rcs_data[batch_indices].unsqueeze(1),
            'tf': self.tf_images[batch_indices],
            'labels': self.rcs_labels[batch_indices]
        }
        return samples2

    def __len__(self):
        return len(self.rcs_data)

    def __getitem__(self, idx):
        # [修正] 这里不能调 super().__getitem__，要写真实的读取逻辑
        
        # ---------- RCS ----------
        rcs = self.rcs_data[idx]         # (256,)
        rcs = rcs.unsqueeze(0)           # → (1,256)
        label = int(self.rcs_labels[idx])

        # ---------- TF ----------
        tf_img = self.tf_images[idx]     # (1,H,W)
        if self.tf_transform:
            tf_img = self.tf_transform(tf_img)

        # 返回 idx 供 Hard Mining 使用
        return rcs, tf_img, label, idx


class Test_RCSTFJointDataset(Dataset):
    """
    RCS和TF联合测试数据集
    """
    def __init__(self, rcs_mat_path, tf_images, tf_labels, tf_transform=None):
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

        # 测试集不需要返回 idx
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
    
    # # 1. 定义增强策略 (仅针对训练集)
    # if split == 'train':
    #     # 针对 Time-Frequency 图的增强
    #     tf_transform = transforms.Compose([
    #         transforms.RandomHorizontalFlip(p=0.5),      # 水平翻转 (增加样本多样性)
    #         transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)), # 轻微旋转和平移
    #         # 注意：不要用垂直翻转，因为频率轴(Y轴)具有物理意义，不能随意颠倒
    #     ])
    # else:
    #     tf_transform = None

    # 2. 传入 dataset
    if split == 'train':
        dataset = RCSTFJointDataset(
            rcs_mat_path=mat_path,
            tf_images=tf_images,
            tf_labels=tf_labels,
            tf_transform=None  # <--- 这里把 None 改成 tf_transform
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
        
        # 测试一个batch
        batch = next(iter(train_loader))
        
        # 兼容性处理
        if len(batch) == 4:
            rcs, tf, labels, idx = batch
        else:
            rcs, tf, labels = batch
            idx = None
        
        print(f"\n✓ 数据格式检查通过:")
        print(f"  - RCS shape: {rcs.shape} (期望: [batch, 1, 256])")
        print(f"  - TF shape: {tf.shape} (期望: [batch, 1, H, W])")
        print(f"  - Labels shape: {labels.shape} (期望: [batch])")
        if idx is not None:
             print(f"  - Indices shape: {idx.shape} (用于 Hard Mining)")
        
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