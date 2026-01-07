"""
日志工具模块 - 统一的日志记录系统
[修改] 移除了全局 config 依赖，改为动态传入 config
"""
import logging
import os
import sys
from datetime import datetime

# [修改] 不要在模块级别导入 config，避免全局变量锁定
# from config import get_config 
# config = get_config() 

def setup_logger(log_dir, config, log_name='train', level=logging.INFO, add_timestamp=True):
    """
    设置日志记录器
    [修改] 增加了 config 参数
    """
    # 创建日志目录
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建logger（使用唯一名称避免冲突）
    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # [修改] 使用传入的 config 获取 noise_level
    noise_str = f"{config.noise_level}dB" if hasattr(config, 'noise_level') else "NA"
    logger_unique_name = f"{log_name}_{noise_str}_{timestamp_str}"
    
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
    
    # 1. 文件handler
    if add_timestamp:
        log_file = os.path.join(log_dir, f'{log_name}_{timestamp_str}.log')
    else:
        log_file = os.path.join(log_dir, f'{log_name}.log')
    
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # 2. 控制台handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    logger.propagate = False
    
    return logger


def log_config(logger, config):
    """记录配置信息"""
    logger.info("="*60)
    logger.info("Configuration:")
    logger.info("="*60)
    
    if hasattr(config, '__dict__'):
        config_dict = vars(config)
    else:
        config_dict = config
    
    for key in sorted(config_dict.keys()):
        value = config_dict[key]
        logger.info(f"  {key}: {value}")
    logger.info("="*60)


def log_model_info(logger, model):
    """记录模型信息"""
    logger.info("="*60)
    logger.info("Model Information:")
    logger.info("="*60)
    
    try:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"  Total parameters: {total_params:,}")
        logger.info(f"  Trainable parameters: {trainable_params:,}")
    except:
        logger.info("  Could not calculate parameters (model might be simpler object)")
        
    logger.info(f"\nModel structure:")
    logger.info(str(model))
    logger.info("="*60)


def log_epoch_summary(logger, epoch, train_metrics, test_metrics=None):
    """记录epoch总结"""
    logger.info("-" * 60)
    
    # 格式化训练指标
    train_str = " | ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" 
                           for k, v in train_metrics.items()])
    logger.info(f"Epoch {epoch:3d} | Train | {train_str}")
    
    # 格式化测试指标
    if test_metrics:
        test_str = " | ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" 
                              for k, v in test_metrics.items()])
        logger.info(f"Epoch {epoch:3d} | Test  | {test_str}")
        
    logger.info("-" * 60)


def log_best_model(logger, epoch, metric_name, metric_value):
    logger.info("="*60)
    logger.info(f"✓ New best model found!")
    logger.info(f"  Epoch: {epoch}")
    logger.info(f"  {metric_name}: {metric_value:.4f}")
    logger.info("="*60)


def log_training_complete(logger, total_time, best_epoch, best_metric):
    logger.info("\n" + "="*60)
    logger.info("Training Complete!")
    logger.info("="*60)
    logger.info(f"  Total time: {total_time/60:.2f} minutes")
    logger.info(f"  Best epoch: {best_epoch}")
    logger.info(f"  Best test accuracy: {best_metric:.2f}%")
    logger.info("="*60)


class LoggerWrapper:
    """日志包装器"""
    def __init__(self, log_dir, config, log_name='train', add_timestamp=True):
        # [修改] 必须传入 config
        self.logger = setup_logger(log_dir, config, log_name, add_timestamp=add_timestamp)
        self.log_dir = log_dir
        self.config = config
    
    def info(self, msg): self.logger.info(msg)
    def warning(self, msg): self.logger.warning(msg)
    def error(self, msg): self.logger.error(msg)
    
    def log_config(self, config): log_config(self.logger, config)
    def log_model(self, model): log_model_info(self.logger, model)
    def log_epoch(self, epoch, train, test=None): log_epoch_summary(self.logger, epoch, train, test)
    def log_best(self, epoch, name, val): log_best_model(self.logger, epoch, name, val)
    def log_complete(self, time, epoch, val): log_training_complete(self.logger, time, epoch, val)