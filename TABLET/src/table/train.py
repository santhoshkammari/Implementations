import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from datasets import load_dataset
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import os
import json
from tqdm import tqdm
import wandb
from pathlib import Path
import argparse
from datetime import datetime

from tablet import Split, Merge, BasicBlock, extract_table_structure, FocalLoss

class TABLETDataset(Dataset):
    """Dataset for TABLET training with PubTabNet OTSL"""
    
    def __init__(self, split='train', max_samples=None):
        # Load dataset
        self.dataset = load_dataset('ds4sd/PubTabNet_OTSL', split=split)
        if max_samples:
            self.dataset = self.dataset.select(range(min(max_samples, len(self.dataset))))
            
        # OTSL vocabulary
        self.otsl_vocab = {'fcel': 0, 'ecel': 1, 'lcel': 2, 'nl': 3}
        self.vocab_size = len(self.otsl_vocab)
        
        # Transform for images
        self.transform = self.get_transform()
        
    def get_transform(self):
        """TABLET preprocessing transform"""
        return transforms.Compose([
            transforms.Lambda(self.tablet_preprocess),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def tablet_preprocess(self, image):
        """Resize to 960x960 with padding"""
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        w, h = image.size
        if w > h:
            new_w = 960
            new_h = int(h * 960 / w)
        else:
            new_h = 960
            new_w = int(w * 960 / h)
        
        image = image.resize((new_w, new_h), Image.BILINEAR)
        canvas = Image.new('RGB', (960, 960), color=(255, 255, 255))
        
        paste_x = (960 - new_w) // 2
        paste_y = (960 - new_h) // 2
        canvas.paste(image, (paste_x, paste_y))
        
        return canvas
    
    def create_split_targets(self, rows, cols):
        """Create binary targets for row/column splits"""
        # Create targets for Split model
        row_targets = torch.zeros(960, 2)  # [H, 2] - binary classification
        col_targets = torch.zeros(960, 2)  # [W, 2] - binary classification
        
        # Mark split positions as positive class (index 1)
        for row_pos in range(0, 960, 960 // (rows + 1)):
            if 0 < row_pos < 960:
                row_targets[row_pos, 1] = 1.0
                row_targets[row_pos, 0] = 0.0
        
        for col_pos in range(0, 960, 960 // (cols + 1)):
            if 0 < col_pos < 960:
                col_targets[col_pos, 1] = 1.0  
                col_targets[col_pos, 0] = 0.0
        
        # All other positions are "no split" (index 0)
        row_targets[:, 0] = (row_targets[:, 1] == 0).float()
        col_targets[:, 0] = (col_targets[:, 1] == 0).float()
        
        return row_targets, col_targets
    
    def otsl_to_grid_tokens(self, otsl, rows, cols):
        """Convert OTSL sequence to grid cell tokens"""
        tokens = []
        otsl_idx = 0
        
        for i in range(rows):
            for j in range(cols):
                if otsl_idx < len(otsl):
                    token = otsl[otsl_idx]
                    if token in self.otsl_vocab:
                        tokens.append(self.otsl_vocab[token])
                    else:
                        tokens.append(self.otsl_vocab['ecel'])  # default
                    otsl_idx += 1
                else:
                    tokens.append(self.otsl_vocab['ecel'])  # default
            
            # Skip 'nl' tokens in OTSL
            while otsl_idx < len(otsl) and otsl[otsl_idx] == 'nl':
                otsl_idx += 1
        
        return torch.tensor(tokens, dtype=torch.long)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        
        # Get image
        image = sample['image']
        image_tensor = self.transform(image)
        
        # Get table structure info
        rows = sample['rows']
        cols = sample['cols'] 
        otsl = sample['otsl']
        
        # Create targets for Split model
        row_targets, col_targets = self.create_split_targets(rows, cols)
        
        # Create targets for Merge model
        merge_targets = self.otsl_to_grid_tokens(otsl, rows, cols)
        
        return {
            'image': image_tensor,
            'row_targets': row_targets,
            'col_targets': col_targets,
            'merge_targets': merge_targets,
            'rows': rows,
            'cols': cols,
            'filename': sample['filename']
        }

class TABLETTrainer:
    """Trainer for TABLET model"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize models
        self.split_model = Split(BasicBlock, [2, 2, 2, 2], fpn_channels=128).to(self.device)
        
        # Merge model will be initialized dynamically based on grid size
        self.merge_model = None
        
        # Loss functions
        self.focal_loss = FocalLoss(alpha=1.0, gamma=2.0)
        self.ce_loss = nn.CrossEntropyLoss()
        
        # Optimizers
        self.split_optimizer = optim.AdamW(self.split_model.parameters(), 
                                         lr=config['learning_rate'], 
                                         weight_decay=config['weight_decay'])
        
        # Schedulers
        self.split_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.split_optimizer, T_max=config['epochs'], eta_min=1e-6)
        
        # Setup checkpointing
        self.checkpoint_dir = Path(config['checkpoint_dir'])
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Metrics tracking
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        
    def save_checkpoint(self, epoch, val_loss, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'split_model_state_dict': self.split_model.state_dict(),
            'split_optimizer_state_dict': self.split_optimizer.state_dict(),
            'split_scheduler_state_dict': self.split_scheduler.state_dict(),
            'val_loss': val_loss,
            'best_val_loss': self.best_val_loss,
            'config': self.config,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            print(f"💾 New best model saved at epoch {epoch} with val_loss: {val_loss:.4f}")
        
        # Keep only last N checkpoints
        self.cleanup_checkpoints(keep_last=3)
    
    def cleanup_checkpoints(self, keep_last=3):
        """Keep only the most recent checkpoints"""
        checkpoints = sorted(self.checkpoint_dir.glob('checkpoint_epoch_*.pt'))
        if len(checkpoints) > keep_last:
            for old_checkpoint in checkpoints[:-keep_last]:
                old_checkpoint.unlink()
    
    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.split_model.load_state_dict(checkpoint['split_model_state_dict'])
        self.split_optimizer.load_state_dict(checkpoint['split_optimizer_state_dict'])
        self.split_scheduler.load_state_dict(checkpoint['split_scheduler_state_dict'])
        
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        
        return checkpoint['epoch']
    
    def train_epoch(self, train_loader, epoch):
        """Train for one epoch"""
        self.split_model.train()
        total_loss = 0
        split_loss_sum = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            images = batch['image'].to(self.device)
            row_targets = batch['row_targets'].to(self.device)
            col_targets = batch['col_targets'].to(self.device)
            
            # Split model forward pass
            self.split_optimizer.zero_grad()
            
            split_outputs = self.split_model(images)
            row_preds = split_outputs['row_splits']  # [B, H, 2]
            col_preds = split_outputs['col_splits']  # [B, W, 2]
            
            # Split loss using focal loss
            row_loss = self.focal_loss(
                row_preds.view(-1, 2), 
                torch.argmax(row_targets, dim=-1).view(-1)
            )
            col_loss = self.focal_loss(
                col_preds.view(-1, 2),
                torch.argmax(col_targets, dim=-1).view(-1)
            )
            
            split_loss = row_loss + col_loss
            
            # Backward pass
            split_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.split_model.parameters(), 1.0)
            self.split_optimizer.step()
            
            # Update metrics
            total_loss += split_loss.item()
            split_loss_sum += split_loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'Split Loss': f'{split_loss.item():.4f}',
                'Avg Loss': f'{total_loss/(batch_idx+1):.4f}'
            })
            
            # Log to wandb
            if self.config.get('use_wandb', False):
                wandb.log({
                    'train/split_loss': split_loss.item(),
                    'train/row_loss': row_loss.item(),
                    'train/col_loss': col_loss.item(),
                    'train/lr': self.split_optimizer.param_groups[0]['lr']
                })
        
        avg_loss = total_loss / len(train_loader)
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
    def validate(self, val_loader, epoch):
        """Validation loop"""
        self.split_model.eval()
        total_loss = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f'Validation {epoch}')
            
            for batch in pbar:
                images = batch['image'].to(self.device)
                row_targets = batch['row_targets'].to(self.device)
                col_targets = batch['col_targets'].to(self.device)
                
                # Split model forward pass
                split_outputs = self.split_model(images)
                row_preds = split_outputs['row_splits']
                col_preds = split_outputs['col_splits']
                
                # Calculate loss
                row_loss = self.focal_loss(
                    row_preds.view(-1, 2),
                    torch.argmax(row_targets, dim=-1).view(-1)
                )
                col_loss = self.focal_loss(
                    col_preds.view(-1, 2),
                    torch.argmax(col_targets, dim=-1).view(-1)
                )
                
                val_loss = row_loss + col_loss
                total_loss += val_loss.item()
                
                pbar.set_postfix({'Val Loss': f'{val_loss.item():.4f}'})
        
        avg_val_loss = total_loss / len(val_loader)
        self.val_losses.append(avg_val_loss)
        
        # Log to wandb
        if self.config.get('use_wandb', False):
            wandb.log({
                'val/loss': avg_val_loss,
                'epoch': epoch
            })
        
        return avg_val_loss
    
    def train(self, train_loader, val_loader):
        """Full training loop"""
        print(f"🚀 Starting training for {self.config['epochs']} epochs")
        print(f"💾 Checkpoints will be saved to: {self.checkpoint_dir}")
        
        start_epoch = 0
        
        # Resume from checkpoint if specified
        if self.config.get('resume_from_checkpoint'):
            start_epoch = self.load_checkpoint(self.config['resume_from_checkpoint'])
            print(f"📂 Resumed from epoch {start_epoch}")
        
        for epoch in range(start_epoch + 1, self.config['epochs'] + 1):
            print(f"\n📊 Epoch {epoch}/{self.config['epochs']}")
            
            # Training
            train_loss = self.train_epoch(train_loader, epoch)
            
            # Validation
            val_loss = self.validate(val_loader, epoch)
            
            # Learning rate scheduling
            self.split_scheduler.step()
            
            # Print epoch summary
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            # Save every N epochs or if best
            if epoch % self.config['save_every'] == 0 or is_best:
                self.save_checkpoint(epoch, val_loss, is_best)
        
        print(f"🎉 Training completed! Best validation loss: {self.best_val_loss:.4f}")

def main():
    parser = argparse.ArgumentParser(description='Train TABLET model')
    parser.add_argument('--config', type=str, default='config.json', help='Config file path')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    parser.add_argument('--debug', action='store_true', help='Debug mode with small dataset')
    args = parser.parse_args()
    
    # Default configuration
    config = {
        'batch_size': 8,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'epochs': 50,
        'save_every': 5,
        'checkpoint_dir': 'checkpoints',
        'use_wandb': False,
        'wandb_project': 'tablet-training',
        'max_train_samples': 1000 if args.debug else None,
        'max_val_samples': 100 if args.debug else None
    }
    
    # Load config from file if exists
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config.update(json.load(f))
    
    # Override with command line args
    if args.resume:
        config['resume_from_checkpoint'] = args.resume
    
    print("🔧 Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Initialize wandb
    if config['use_wandb']:
        wandb.init(
            project=config['wandb_project'],
            config=config,
            name=f"tablet_train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
    
    # Create datasets
    print("📚 Loading datasets...")
    train_dataset = TABLETDataset('train', max_samples=config['max_train_samples'])
    val_dataset = TABLETDataset('validation', max_samples=config['max_val_samples'])
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize trainer
    trainer = TABLETTrainer(config)
    
    # Start training
    trainer.train(train_loader, val_loader)
    
    # Save final config
    with open(trainer.checkpoint_dir / 'final_config.json', 'w') as f:
        json.dump(config, f, indent=2)

if __name__ == "__main__":
    main()