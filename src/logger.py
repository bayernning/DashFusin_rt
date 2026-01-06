"""
日志工具模块 - 统一的日志记录系统
支持同时输出到文件和控制台
"""
import logging
import os
import sys
from datetime import datetime
from config import get_config


config = get_config()

def setup_logger(log_dir, log_name='train', level=logging.INFO, add_timestamp=True):
    """
    设置日志记录器
    
    Args:
        log_dir: 日志文件保存目录
        log_name: 日志文件名（不含扩展名）
        level: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        add_timestamp: 是否在文件名中添加时间戳（避免覆盖）
    
    Returns:
        logger: 配置好的日志记录器
    """
    # 创建日志目录
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建logger（使用唯一名称避免冲突）
    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    logger_unique_name = f"{log_name}_{config.noise_level}dB_{timestamp_str}"
    logger = logging.getLogger(logger_unique_name)
    logger.setLevel(level)
    
    # 清除已有的handlers（避免重复）
    if logger.handlers:
        logger.handlers.clear()
    
    # 日志格式
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 1. 文件handler - 保存所有日志
    if add_timestamp:
        log_file = os.path.join(log_dir, f'{log_name}_{timestamp_str}.log')
    else:
        log_file = os.path.join(log_dir, f'{log_name}.log')
    
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # 2. 控制台handler - 同时输出到控制台
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 防止日志向上传播到root logger
    logger.propagate = False
    
    return logger


def log_config(logger, config):
    """
    记录配置信息
    
    Args:
        logger: 日志记录器
        config: 配置对象
    """
    logger.info("="*60)
    logger.info("Configuration:")
    logger.info("="*60)
    
    # 将config转为字典（如果是argparse.Namespace）
    if hasattr(config, '__dict__'):
        config_dict = vars(config)
    else:
        config_dict = config
    
    # 按键排序输出
    for key in sorted(config_dict.keys()):
        value = config_dict[key]
        logger.info(f"  {key}: {value}")
    
    logger.info("="*60)


def log_model_info(logger, model):
    """
    记录模型信息
    
    Args:
        logger: 日志记录器
        model: PyTorch模型
    """
    logger.info("="*60)
    logger.info("Model Information:")
    logger.info("="*60)
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")
    logger.info(f"  Non-trainable parameters: {total_params - trainable_params:,}")
    
    # 模型结构
    logger.info(f"\nModel structure:")
    logger.info(str(model))
    logger.info("="*60)


def log_metrics(logger, epoch, metrics_dict, prefix='Train'):
    """
    记录训练/测试指标
    
    Args:
        logger: 日志记录器
        epoch: 当前epoch
        metrics_dict: 指标字典 {'loss': 0.5, 'acc': 0.85, ...}
        prefix: 前缀 ('Train' 或 'Test')
    """
    metrics_str = " | ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" 
                               for k, v in metrics_dict.items()])
    logger.info(f"Epoch {epoch:3d} | {prefix:5s} | {metrics_str}")


def log_epoch_summary(logger, epoch, train_metrics, test_metrics=None):
    """
    记录epoch总结
    
    Args:
        logger: 日志记录器
        epoch: 当前epoch
        train_metrics: 训练指标字典
        test_metrics: 测试指标字典（可选）
    """
    logger.info("-"*60)
    log_metrics(logger, epoch, train_metrics, prefix='Train')
    if test_metrics is not None:
        log_metrics(logger, epoch, test_metrics, prefix='Test')
    logger.info("-"*60)


def log_best_model(logger, epoch, metric_name, metric_value):
    """
    记录最佳模型
    
    Args:
        logger: 日志记录器
        epoch: 当前epoch
        metric_name: 指标名称
        metric_value: 指标值
    """
    logger.info("="*60)
    logger.info(f"✓ New best model found!")
    logger.info(f"  Epoch: {epoch}")
    logger.info(f"  {metric_name}: {metric_value:.4f}")
    logger.info("="*60)


def log_training_complete(logger, total_time, best_epoch, best_metric):
    """
    记录训练完成
    
    Args:
        logger: 日志记录器
        total_time: 总训练时间（秒）
        best_epoch: 最佳epoch
        best_metric: 最佳指标值
    """
    logger.info("\n" + "="*60)
    logger.info("Training Complete!")
    logger.info("="*60)
    logger.info(f"  Total time: {total_time/60:.2f} minutes")
    logger.info(f"  Best epoch: {best_epoch}")
    logger.info(f"  Best test accuracy: {best_metric:.2f}%")
    logger.info("="*60)


# ============= 便捷函数 =============

class LoggerWrapper:
    """
    日志包装器 - 方便使用
    """
    def __init__(self, log_dir, log_name='train', add_timestamp=True):
        self.logger = setup_logger(log_dir, log_name, add_timestamp=add_timestamp)
        self.log_dir = log_dir
        self.log_name = log_name
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 记录日志文件路径，方便后续查看
        if add_timestamp:
            self.log_file = os.path.join(log_dir, f'{log_name}_{self.timestamp}.log')
        else:
            self.log_file = os.path.join(log_dir, f'{log_name}.log')
    
    def get_log_file(self):
        """获取日志文件路径"""
        return self.log_file
    
    def info(self, msg):
        self.logger.info(msg)
    
    def warning(self, msg):
        self.logger.warning(msg)
    
    def error(self, msg):
        self.logger.error(msg)
    
    def debug(self, msg):
        self.logger.debug(msg)
    
    def log_config(self, config):
        log_config(self.logger, config)
    
    def log_model(self, model):
        log_model_info(self.logger, model)
    
    def log_epoch(self, epoch, train_metrics, test_metrics=None):
        log_epoch_summary(self.logger, epoch, train_metrics, test_metrics)
    
    def log_best(self, epoch, metric_name, metric_value):
        log_best_model(self.logger, epoch, metric_name, metric_value)
    
    def log_complete(self, total_time, best_epoch, best_metric):
        log_training_complete(self.logger, total_time, best_epoch, best_metric)


# ============= 测试代码 =============

if __name__ == '__main__':
    # 测试日志系统
    import argparse
    
    # 创建测试配置
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    config = parser.parse_args([])
    
    # 创建日志
    logger = LoggerWrapper('./test_logs', 'test')
    
    # 测试各种日志功能
    logger.info("Testing logger...")
    logger.log_config(config)
    
    # 模拟训练
    import torch.nn as nn
    model = nn.Sequential(
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10)
    )
    logger.log_model(model)
    
    # 模拟epoch
    for epoch in range(1, 4):
        train_metrics = {
            'loss': 2.5 - epoch*0.3,
            'acc': 30 + epoch*10,
            'lr': 0.001
        }
        test_metrics = {
            'loss': 2.3 - epoch*0.2,
            'acc': 35 + epoch*8
        }
        logger.log_epoch(epoch, train_metrics, test_metrics)
        
        if epoch == 2:
            logger.log_best(epoch, 'test_acc', test_metrics['acc'])
    
    logger.log_complete(total_time=150, best_epoch=2, best_metric=51.0)
    
    print("\n✓ Logger test complete! Check ./test_logs/test.log")
