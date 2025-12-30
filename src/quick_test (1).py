"""
快速测试脚本 - 验证所有模块是否正常工作
"""
import torch
import sys
import os

print("="*60)
print("DashFusion Quick Test")
print("="*60)

# 测试1: 导入模块
print("\n[Test 1/7] Testing module imports...")
try:
    from config import get_config
    from model.layers import TransformerEncoderLayer, CrossModalAttention, HierarchicalBottleneckFusion
    from model.encoders import RCSEncoder, JTFEncoder
    
    from model.MLP import Projector, MultimodalClassifier, DualProjector
    from model.dashfusion import DashFusion
    from dataloader.dataset import get_dataloader, create_dummy_data
    from train import Trainer
    from utils import set_seed
    print("✓ All modules imported successfully")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

# 测试2: 配置
print("\n[Test 2/8] Testing configuration...")
try:
    config = get_config()
    config.batch_size = 4
    config.epochs = 2
    config.num_workers = 0
    print(f"✓ Configuration loaded")
    print(f"  - Device: {config.device}")
    print(f"  - Hidden dim: {config.hidden_dim}")
    print(f"  - Num classes: {config.num_classes}")
except Exception as e:
    print(f"✗ Configuration error: {e}")
    sys.exit(1)

# 测试3: MLP模块
print("\n[Test 3/8] Testing MLP modules...")
try:
    # 测试Projector
    projector = Projector(input_dim=128, output_dim=64)
    test_feat = torch.randn(4, 128)
    proj_out = projector(test_feat)
    assert proj_out.shape == (4, 64), "Projector output shape mismatch"
    # 检查L2归一化
    norms = torch.norm(proj_out, p=2, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), "L2 normalization failed"
    print("  ✓ Projector working")
    
    # 测试DualProjector
    dual_proj = DualProjector(rcs_dim=128, jtf_dim=128, proj_dim=64)
    rcs_feat = torch.randn(4, 128)
    jtf_feat = torch.randn(4, 128)
    rcs_proj, jtf_proj = dual_proj(rcs_feat, jtf_feat)
    assert rcs_proj.shape == (4, 64) and jtf_proj.shape == (4, 64)
    print("  ✓ DualProjector working")
    
    # 测试MultimodalClassifier
    mm_clf = MultimodalClassifier(
        rcs_dim=128, jtf_dim=128, bottleneck_dim=128,
        hidden_dims=[256, 128], num_classes=10
    )
    logits = mm_clf(rcs_feat, jtf_feat, torch.randn(4, 128))
    assert logits.shape == (4, 10), "Classifier output shape mismatch"
    print("  ✓ MultimodalClassifier working")
    
    print("✓ All MLP modules working correctly")
except Exception as e:
    print(f"✗ MLP module error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试4: 数据生成
print("\n[Test 4/8] Testing data generation...")
try:
    if not os.path.exists('./dataset/train_rcs.npy'):
        create_dummy_data('./dataset/', num_train=100, num_val=20, num_test=20, 
                         num_classes=config.num_classes)
    print("✓ Dataset created/verified")
except Exception as e:
    print(f"✗ Data generation error: {e}")
    sys.exit(1)

# 测试5: 数据加载
print("\n[Test 5/8] Testing data loading...")
try:
    train_loader = get_dataloader(config, split='train')
    val_loader = get_dataloader(config, split='val')
    
    # 测试一个batch
    rcs, jtf, labels = next(iter(train_loader))
    print(f"✓ Data loader working")
    print(f"  - RCS shape: {rcs.shape}")
    print(f"  - JTF shape: {jtf.shape}")
    print(f"  - Labels shape: {labels.shape}")
    
    assert rcs.shape == (config.batch_size, 1, 256), "RCS shape mismatch"
    assert jtf.shape == (config.batch_size, 1, 256, 256), "JTF shape mismatch"
    assert labels.shape == (config.batch_size,), "Labels shape mismatch"
    print("✓ Data shapes verified")
    
except Exception as e:
    print(f"✗ Data loading error: {e}")
    sys.exit(1)

# 测试6: 模型构建
print("\n[Test 6/8] Testing model building...")
try:
    device = torch.device(config.device)
    model = DashFusion(config).to(device)
    
    num_params = model.get_num_params()
    print(f"✓ Model built successfully")
    print(f"  - Total parameters: {num_params:,}")
    
except Exception as e:
    print(f"✗ Model building error: {e}")
    sys.exit(1)

# 测试7: 前向传播
print("\n[Test 7/8] Testing forward pass...")
try:
    model.eval()
    rcs = rcs.to(device)
    jtf = jtf.to(device)
    labels = labels.to(device)
    
    # 测试无标签前向传播
    with torch.no_grad():
        outputs = model(rcs, jtf)
        assert 'logits' in outputs, "Missing logits in output"
        print(f"✓ Forward pass (inference) successful")
        print(f"  - Output logits shape: {outputs['logits'].shape}")
    
    # 测试有标签前向传播(训练模式)
    model.train()
    outputs = model(rcs, jtf, labels)
    assert 'loss' in outputs, "Missing loss in output"
    assert 'cls_loss' in outputs, "Missing cls_loss in output"
    assert 'contrast_loss' in outputs, "Missing contrast_loss in output"
    
    print(f"✓ Forward pass (training) successful")
    print(f"  - Total loss: {outputs['loss'].item():.4f}")
    print(f"  - Classification loss: {outputs['cls_loss'].item():.4f}")
    print(f"  - Contrast loss: {outputs['contrast_loss'].item():.4f}")
    
    # 测试反向传播
    loss = outputs['loss']
    loss.backward()
    print(f"✓ Backward pass successful")
    
except Exception as e:
    print(f"✗ Forward/Backward pass error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试8: 快速训练测试
print("\n[Test 8/8] Testing training loop (1 epoch)...")
try:
    config.epochs = 1
    trainer = Trainer(model, train_loader, val_loader, config)
    
    # 训练一个epoch
    train_loss, cls_loss, con_loss, train_acc = trainer.train_epoch(1)
    print(f"✓ Training epoch completed")
    print(f"  - Train loss: {train_loss:.4f}")
    print(f"  - Train accuracy: {train_acc:.2f}%")
    
    # 验证
    val_loss, val_acc = trainer.validate()
    print(f"✓ Validation completed")
    print(f"  - Val loss: {val_loss:.4f}")
    print(f"  - Val accuracy: {val_acc:.2f}%")
    
except Exception as e:
    print(f"✗ Training loop error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 全部测试通过
print("\n" + "="*60)
print("✓ All tests passed!")
print("="*60)
print("\nYou can now run: python main.py")
print("="*60)
