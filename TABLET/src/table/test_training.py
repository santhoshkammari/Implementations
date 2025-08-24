#!/usr/bin/env python3
"""
Quick test script to verify training setup works
"""

import torch
from train import TABLETDataset, TABLETTrainer
import json

def test_dataset():
    """Test dataset loading"""
    print("🧪 Testing dataset loading...")
    
    # Load small dataset
    dataset = TABLETDataset('train', max_samples=5)
    print(f"Dataset size: {len(dataset)}")
    
    # Test one sample
    sample = dataset[0]
    print(f"Image shape: {sample['image'].shape}")
    print(f"Row targets shape: {sample['row_targets'].shape}")
    print(f"Col targets shape: {sample['col_targets'].shape}")
    print(f"Merge targets shape: {sample['merge_targets'].shape}")
    print(f"Grid size: {sample['rows']} x {sample['cols']}")
    print(f"Filename: {sample['filename']}")
    
    return dataset

def test_model_forward():
    """Test model forward pass"""
    print("\n🧪 Testing model forward pass...")
    
    from tablet import Split, BasicBlock
    
    # Create model
    model = Split(BasicBlock, [2, 2, 2, 2], fpn_channels=128)
    
    # Test input
    x = torch.randn(1, 3, 960, 960)
    
    with torch.no_grad():
        outputs = model(x)
        print(f"Row splits shape: {outputs['row_splits'].shape}")
        print(f"Col splits shape: {outputs['col_splits'].shape}")
    
    return model

def test_training_step():
    """Test one training step"""
    print("\n🧪 Testing training step...")
    
    # Simple config
    config = {
        'batch_size': 2,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'epochs': 1,
        'save_every': 1,
        'checkpoint_dir': 'test_checkpoints',
        'use_wandb': False,
        'max_train_samples': 5,
        'max_val_samples': 2
    }
    
    # Create datasets
    train_dataset = TABLETDataset('train', max_samples=5)
    val_dataset = TABLETDataset('validation', max_samples=2)
    
    # Create data loaders
    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
    
    # Create trainer
    trainer = TABLETTrainer(config)
    
    # Test one training step
    try:
        batch = next(iter(train_loader))
        print(f"Batch image shape: {batch['image'].shape}")
        print(f"Batch row targets shape: {batch['row_targets'].shape}")
        
        # Test training step
        train_loss = trainer.train_epoch(train_loader, epoch=1)
        print(f"Training completed! Loss: {train_loss:.4f}")
        
        # Test validation step
        val_loss = trainer.validate(val_loader, epoch=1)
        print(f"Validation completed! Loss: {val_loss:.4f}")
        
        print("✅ Training test successful!")
        
    except Exception as e:
        print(f"❌ Training test failed: {e}")
        raise

if __name__ == "__main__":
    print("🚀 TABLET Training Test Suite")
    print("=" * 50)
    
    # Test dataset
    dataset = test_dataset()
    
    # Test model
    model = test_model_forward()
    
    # Test training
    test_training_step()
    
    print("\n✅ All tests passed! Ready to start training.")
    print("\n📖 To start full training:")
    print("   python train.py --config config.json")
    print("   python train.py --config config.json --debug  # For quick test")
    print("   python train.py --resume checkpoints/best_model.pt  # Resume training")