import torch
import torch.nn as nn
import torch.nn.functional as F


from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange, Reduce
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

H,W = 960,960


class FocalLoss(nn.Module):
    """Focal Loss as used in TABLET"""
    def __init__(self, alpha=1.0, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs.view(-1, inputs.size(-1)), targets.view(-1), reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

def inference(model):
    """Inference function for TABLET table structure recognition"""
    model.eval()
    print("TABLET model created, ready to test...")
    
    # TABLET preprocessing - resize to 960x960 with padding as described in paper
    def tablet_transform(image):
        """
        TABLET preprocessing: resize longer side to 960 pixels, 
        maintain aspect ratio, then pad to 960x960
        """
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Get original dimensions
        w, h = image.size
        
        # Resize longer side to 960 while maintaining aspect ratio
        if w > h:
            new_w = 960
            new_h = int(h * 960 / w)
        else:
            new_h = 960
            new_w = int(w * 960 / h)
        
        # Resize image
        image = image.resize((new_w, new_h), Image.BILINEAR)
        
        # Create 960x960 canvas and paste resized image
        canvas = Image.new('RGB', (960, 960), color=(255, 255, 255))  # white background
        
        # Center the image on canvas
        paste_x = (960 - new_w) // 2
        paste_y = (960 - new_h) // 2
        canvas.paste(image, (paste_x, paste_y))
        
        # Convert to tensor and normalize
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        return transform(canvas)
    
    # Load your table image
    img_path = "/home/ntlpt59/Pictures/Screenshots/Screenshot from 2025-05-24 01-17-32.png"
    img = Image.open(img_path)
    
    print(f"Original image size: {img.size}")
    print(f"Image mode: {img.mode}")
    
    # Apply TABLET preprocessing
    img_tensor = tablet_transform(img).unsqueeze(0)  # add batch dimension
    
    print(f"Preprocessed tensor shape: {img_tensor.shape}")  # Should be model = Split(BasicBlock, [2,2,2,2], fpn_channels=128)
    
    # Run inference
    with torch.no_grad():
        outputs = model(img_tensor)
        
        print("\n=== TABLET Outputs ===")
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                print(f"{key}: {value.shape}")
            else:
                print(f"{key}: {type(value)}")
        
        # Extract split predictions
        row_splits = outputs['row_splits']  # [1, H, 2] - binary classification
        col_splits = outputs['col_splits']  # [1, W, 2] - binary classification
        
        # Get split probabilities (softmax)
        row_probs = F.softmax(row_splits, dim=-1)[0, :, 1]  # [H] - probability of split
        col_probs = F.softmax(col_splits, dim=-1)[0, :, 1]  # [W] - probability of split
        
        print(f"\nRow split probabilities shape: {row_probs.shape}")
        print(f"Column split probabilities shape: {col_probs.shape}")
        
        # Get binary split decisions (threshold at 0.5)
        row_split_mask = row_probs > 0.5
        col_split_mask = col_probs > 0.5
        
        print(f"Number of row splits detected: {row_split_mask.sum().item()}")
        print(f"Number of column splits detected: {col_split_mask.sum().item()}")
        
        # Find split positions
        row_split_positions = torch.where(row_split_mask)[0].cpu().numpy()
        col_split_positions = torch.where(col_split_mask)[0].cpu().numpy()
        
        print(f"Row split positions: {row_split_positions}")
        print(f"Column split positions: {col_split_positions}")
        
        # Visualize results
        visualize_splits(img, row_split_positions, col_split_positions, 
                        row_probs.cpu().numpy(), col_probs.cpu().numpy())
        
        return outputs

def visualize_splits(original_image, row_positions, col_positions, row_probs, col_probs):
    """Visualize the detected table splits"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Original image
    axes[0, 0].imshow(original_image)
    axes[0, 0].set_title('Original Table Image')
    axes[0, 0].axis('off')
    
    # Image with detected splits
    img_array = np.array(original_image)
    axes[0, 1].imshow(img_array)
    
    # Scale positions to original image size
    img_h, img_w = img_array.shape[:2]
    
    # Draw row splits (horizontal lines)
    for pos in row_positions:
        y = int(pos * img_h / 960)  # scale back to original size
        axes[0, 1].axhline(y=y, color='red', linewidth=2, alpha=0.7)
    
    # Draw column splits (vertical lines)  
    for pos in col_positions:
        x = int(pos * img_w / 960)  # scale back to original size
        axes[0, 1].axvline(x=x, color='blue', linewidth=2, alpha=0.7)
    
    axes[0, 1].set_title(f'Detected Splits\n({len(row_positions)} rows, {len(col_positions)} cols)')
    axes[0, 1].axis('off')
    
    # Row split probabilities
    axes[1, 0].plot(row_probs)
    axes[1, 0].axhline(y=0.5, color='red', linestyle='--', alpha=0.7)
    axes[1, 0].set_title('Row Split Probabilities')
    axes[1, 0].set_xlabel('Pixel Position (Height)')
    axes[1, 0].set_ylabel('Split Probability')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Column split probabilities
    axes[1, 1].plot(col_probs)
    axes[1, 1].axhline(y=0.5, color='blue', linestyle='--', alpha=0.7)
    axes[1, 1].set_title('Column Split Probabilities')
    axes[1, 1].set_xlabel('Pixel Position (Width)')
    axes[1, 1].set_ylabel('Split Probability')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def extract_table_structure(row_positions, col_positions):
    """
    Extract table structure from split positions
    Returns grid coordinates for further processing
    """
    # Add boundaries (0 and max position)
    row_boundaries = np.concatenate([[0], row_positions, [960]])
    col_boundaries = np.concatenate([[0], col_positions, [960]])
    
    # Sort boundaries
    row_boundaries = np.sort(row_boundaries)
    col_boundaries = np.sort(col_boundaries)
    
    # Create grid structure
    grid = []
    for i in range(len(row_boundaries) - 1):
        row_cells = []
        for j in range(len(col_boundaries) - 1):
            cell = {
                'row_start': row_boundaries[i],
                'row_end': row_boundaries[i + 1],
                'col_start': col_boundaries[j], 
                'col_end': col_boundaries[j + 1],
                'row_idx': i,
                'col_idx': j
            }
            row_cells.append(cell)
        grid.append(row_cells)
    
    print(f"\nExtracted grid structure: {len(grid)} rows × {len(grid[0])} columns")
    
    return grid

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.main_path = nn.Sequential(
            nn.Conv2d(inplanes, planes, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, 3, padding=1, bias=False),
            nn.BatchNorm2d(planes),
        )
        self.downsample = downsample
        
    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        out = self.main_path(x)
        return F.relu(out + residual)

def make_layer(inplanes, planes, block, n_blocks, stride=1):
    downsample = None
    if stride != 1 or inplanes != planes * block.expansion:
        # output size won't match input, so adjust residual
        downsample = nn.Sequential(
            nn.Conv2d(inplanes, planes * block.expansion,
                     kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes * block.expansion),
        )
    return nn.Sequential(
        block(inplanes, planes, stride, downsample),
        *[block(planes * block.expansion, planes) for _ in range(1, n_blocks)]
    )

class EinopsFPN(nn.Module):
    def __init__(self, in_channels_list, out_channels=128):
        super().__init__()
        # lateral connections - project each stage to same channel count
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, 1, bias=False) 
            for in_ch in in_channels_list
        ])
        
        # smooth the upsampled features
        self.fpn_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
            for _ in in_channels_list
        ])
        
    def forward(self, features):
        # features is list [C1, C2, C3, C4] from different stages
        laterals = [lateral_conv(feat) for lateral_conv, feat in zip(self.lateral_convs, features)]
        
        # top-down pathway using einops for upsampling
        for i in range(len(laterals) - 2, -1, -1):
            higher_feat = laterals[i+1]
            current_feat = laterals[i]
            
            # upsample using repeat - cleaner than interpolate
            b, c, h, w = current_feat.shape
            upsampled = repeat(higher_feat, 'b c h w -> b c (h h2) (w w2)', h2=2, w2=2)
            
            # crop if needed (in case of odd dimensions)
            upsampled = upsampled[:, :, :h, :w]
            
            laterals[i] = current_feat + upsampled
            
        # apply final convs
        fpn_outs = [fpn_conv(lateral) for fpn_conv, lateral in zip(self.fpn_convs, laterals)]
        
        return fpn_outs[0]  # return the highest resolution feature (H/2 x W/2)

class ResNetFPN(nn.Module):
    def __init__(self, block, layers, fpn_channels=128):
        super().__init__()
        
        # stem with einops rearrange
        self.stem = nn.Sequential(
            Rearrange('b c h w -> b c h w'),  # explicit input shape
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # resnet stages
        self.layer1 = make_layer(32, 32, block, layers[0], stride=1)      # H/2, 32 channels
        self.layer2 = make_layer(32, 64, block, layers[1], stride=2)      # H/4, 64 channels  
        self.layer3 = make_layer(64, 128, block, layers[2], stride=2)     # H/8, 128 channels
        self.layer4 = make_layer(128, 256, block, layers[3], stride=2)    # H/16, 256 channels
        
        # FPN
        self.fpn = EinopsFPN([32, 64, 128, 256], fpn_channels)
        
        # init weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                
    def forward(self, x):
        # stem
        x = self.stem(x)
        
        # collect features from each stage
        c1 = self.layer1(x)    # H/2 x W/2 x 32
        c2 = self.layer2(c1)   # H/4 x W/4 x 64
        c3 = self.layer3(c2)   # H/8 x W/8 x 128  
        c4 = self.layer4(c3)   # H/16 x W/16 x 256
        
        # FPN processes all scales, returns H/2 x W/2 x 128
        fpn_out = self.fpn([c1, c2, c3, c4])
        
        return fpn_out

class GlobalProjection(nn.Module):
    """Extract global features using learnable weighted averages as described in TABLET"""
    def __init__(self, fpn_channels):
        super().__init__()
        # Learnable parameters for weighted averaging
        # For horizontal projection: average W/2 values -> output H/2 × fpn_channels
        self.horizontal_weights = nn.Parameter(torch.ones(1, fpn_channels, 1, 1))
        
        # For vertical projection: average H/2 values -> output fpn_channels × W/2  
        self.vertical_weights = nn.Parameter(torch.ones(1, fpn_channels, 1, 1))
        
    def forward(self, x):
        # x shape: [batch, fpn_channels, H/2, W/2]
        batch, channels, h, w = x.shape
        
        # Horizontal projection: learnable weighted average along width dimension
        # Apply learnable weights then global average pool along width
        weighted_x = x * self.horizontal_weights
        FRG = torch.mean(weighted_x, dim=3)  # [batch, fpn_channels, H/2]
        FRG = FRG.permute(0, 2, 1)  # [batch, H/2, fpn_channels] -> [batch, H/2, 128]
        
        # Vertical projection: learnable weighted average along height dimension
        weighted_x = x * self.vertical_weights  
        FCG = torch.mean(weighted_x, dim=2)  # [batch, fpn_channels, W/2]
        FCG = FCG.permute(0, 2, 1)  # [batch, W/2, fpn_channels] -> [batch, W/2, 128]
        FCG = FCG.permute(0, 2, 1)  # -> [batch, 128, W/2] as per paper
        
        return FRG, FCG

class LocalFeatureExtraction(nn.Module):
    """Extract local features as described in TABLET Section 3.1"""
    def __init__(self, fpn_channels):
        super().__init__()
        # For horizontal: 1×2 AvgPool + 1×1 Conv for dimensionality reduction
        self.horizontal_pool = nn.AvgPool2d(kernel_size=(1, 2))
        self.horizontal_conv = nn.Conv2d(fpn_channels, 1, kernel_size=1)
        
        # For vertical: 2×1 AvgPool + 1×1 Conv for dimensionality reduction  
        self.vertical_pool = nn.AvgPool2d(kernel_size=(2, 1))
        self.vertical_conv = nn.Conv2d(fpn_channels, 1, kernel_size=1)
        
    def forward(self, x):
        # x shape: [batch, fpn_channels, H/2, W/2]
        
        # Horizontal local features: H/2 × W/4
        FRL = self.horizontal_pool(x)  # [batch, 128, H/2, W/4]
        FRL = self.horizontal_conv(FRL)  # [batch, 1, H/2, W/4]
        
        # Vertical local features: H/4 × W/2
        FCL = self.vertical_pool(x)  # [batch, 128, H/4, W/2] 
        FCL = self.vertical_conv(FCL)  # [batch, 1, H/4, W/2]
        
        return FRL, FCL


class Split(nn.Module):
    """TABLET Split Model following the exact architecture from the paper"""
    def __init__(self, block, layers, fpn_channels=128):
        super().__init__()
        # Modified ResNet-FPN: remove MaxPool & halve channels
        self.rfpn = ResNetFPN(block, layers, fpn_channels)  # Output: [batch, 128, H/2, W/2]
        
        # Global and local feature extraction
        self.global_projection = GlobalProjection(fpn_channels)
        self.local_extraction = LocalFeatureExtraction(fpn_channels)
        
        # Transformer encoders for row and column splitting
        # Input dimensions based on paper: H/2 × (128 + W/4) and W/2 × (128 + H/4)
        self.row_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=fpn_channels + W//4,  # 128 + W/4 (assuming W=960, W/4=240)
                nhead=8,
                dim_feedforward=2048,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=3
        )
        
        self.col_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=fpn_channels + 240,  # 128 + H/4 (assuming H=960, H/4=240) 
                nhead=8,
                dim_feedforward=2048,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=3
        )
        
        # 1D positional embeddings
        self.row_pos_embed = nn.Parameter(torch.randn(H//2, fpn_channels + W//4))  # H/2 = 480
        self.col_pos_embed = nn.Parameter(torch.randn(W//2, fpn_channels + H//4))  # W/2 = 480
        
        # Classification heads for binary split/no-split
        self.row_classifier = nn.Linear(fpn_channels + W//4, 2)
        self.col_classifier = nn.Linear(fpn_channels + H//4, 2)
        
        # Focal loss
        self.focal_loss = FocalLoss(alpha=1.0, gamma=2.0)
        
    def forward(self, x):
        # Extract FPN features: F1/2 of size H/2 × W/2 × 128
        fpn_features = self.rfpn(x)  # [batch, 128, H/2, W/2]
        print(f'{fpn_features.shape=}')
        
        # Extract global features
        FRG, FCG = self.global_projection(fpn_features)
        # FRG: [batch, H/2, 128], FCG: [batch, 128, W/2]
        print(f'{FRG.shape=}', f'{FCG.shape=}')
        
        # Extract local features  
        FRL, FCL = self.local_extraction(fpn_features)
        # FRL: [batch, 1, H/2, W/4], FCL: [batch, 1, H/4, W/2] (after fix)
        print(f'{FRL.shape=}', f'{FCL.shape=}')
        
        # Prepare horizontal features: FRG + FRL -> H/2 × (128 + W/4)
        batch, _, h_half, w_quarter = FRL.shape
        FRL_flattened = FRL.squeeze(1)  # [batch, 1, H/2, W/4] -> [batch, H/2, W/4]
        FRG_L = torch.cat([FRG, FRL_flattened], dim=-1)  # [batch, H/2, 128 + W/4]
        print(f'{FRG_L.shape=}')

        # Prepare vertical features: FCG + FCL -> W/2 × (128 + H/4)  
        batch, _, h_quarter, w_half = FCL.shape
        FCL_flattened = FCL.squeeze(1).permute(0, 2, 1)  # [batch, 1, H/4, W/2] -> [batch, W/2, H/4]
        FCG_transposed = FCG.permute(0, 2, 1)  # [batch, W/2, 128]
        FCG_L = torch.cat([FCG_transposed, FCL_flattened], dim=-1)  # [batch, W/2, 128 + H/4]
        print(f'{FCG_L.shape=}')
        
        # Add positional embedding and apply transformer
        FRG_L = FRG_L + self.row_pos_embed[:h_half].unsqueeze(0)
        FR = self.row_transformer(FRG_L)  # [batch, H/2, 128 + W/4]
        print(f'{FR.shape=}')
        
        # Add positional embedding and apply transformer
        FCG_L = FCG_L + self.col_pos_embed[:w_half].unsqueeze(0)
        FC = self.col_transformer(FCG_L)  # [batch, W/2, 128 + H/4]
        print(f'{FC.shape=}')
        
        # Binary classification for split/no-split
        row_splits = self.row_classifier(FR)  # [batch, H/2, 2]
        col_splits = self.col_classifier(FC)  # [batch, W/2, 2]
        
        # Upsample 2x to match input image resolution
        row_splits_upsampled = F.interpolate(
            row_splits.permute(0, 2, 1), 
            scale_factor=2, 
            mode='nearest'
        ).permute(0, 2, 1)  # [batch, H, 2]
        
        col_splits_upsampled = F.interpolate(
            col_splits.permute(0, 2, 1),
            scale_factor=2,
            mode='nearest'  
        ).permute(0, 2, 1)  # [batch, W, 2]
        
        return {
            'fpn_features': fpn_features,
            'global_horizontal': FRG,  # [batch, H/2, 128]
            'global_vertical': FCG,    # [batch, 128, W/2] 
            'row_splits': row_splits_upsampled,  # [batch, H, 2]
            'col_splits': col_splits_upsampled,  # [batch, W, 2]
            'row_features': FR,        # [batch, H/2, 128 + W/4]
            'col_features': FC         # [batch, W/2, 128 + H/4]
        }

class MergeResNetFPN(nn.Module):
    """Modified ResNet-FPN for Merge model - outputs 256 channels at H/4 x W/4"""
    def __init__(self, block, layers, fpn_channels=256):
        super().__init__()
        
        # stem with einops rearrange - same as Split but different output dims
        self.stem = nn.Sequential(
            Rearrange('b c h w -> b c h w'),  # explicit input shape
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # Additional pooling for H/4, W/4
        )
        
        # resnet stages - modified for H/4 output
        self.layer1 = make_layer(32, 64, block, layers[0], stride=1)      # H/4, 64 channels
        self.layer2 = make_layer(64, 128, block, layers[1], stride=2)     # H/8, 128 channels  
        self.layer3 = make_layer(128, 256, block, layers[2], stride=2)    # H/16, 256 channels
        self.layer4 = make_layer(256, 512, block, layers[3], stride=2)    # H/32, 512 channels
        
        # FPN - returns features at H/4 resolution with 256 channels
        self.fpn = EinopsFPN([64, 128, 256, 512], fpn_channels)
        
        # init weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                
    def forward(self, x):
        # stem - outputs H/4 x W/4 after stem
        x = self.stem(x)
        
        # collect features from each stage  
        c1 = self.layer1(x)    # H/4 x W/4 x 64
        c2 = self.layer2(c1)   # H/8 x W/8 x 128
        c3 = self.layer3(c2)   # H/16 x W/16 x 256  
        c4 = self.layer4(c3)   # H/32 x W/32 x 512
        
        # FPN processes all scales, returns H/4 x W/4 x 256
        fpn_out = self.fpn([c1, c2, c3, c4])
        
        return fpn_out

class Merge(nn.Module):
    """TABLET Merge Model following the exact architecture from the paper"""
    def __init__(self, block, layers, fpn_channels=256, num_rows=10, num_cols=10):
        super().__init__()
        # Modified ResNet-FPN for Merge: outputs 256 channels at H/4 x W/4
        self.rfpn = MergeResNetFPN(block, layers, fpn_channels)
        
        self.num_rows = num_rows
        self.num_cols = num_cols
        
        # ROI Align for extracting cell features
        from torchvision.ops import roi_align
        self.roi_align = roi_align
        self.roi_output_size = (7, 7)  # Standard ROI pooling size
        self.spatial_scale = 0.25  # H/4, W/4 scale
        
        # Flatten & MLP for processing cell features
        self.flatten = nn.Flatten()  # Flatten 7x7x256 -> 12544
        self.mlp = nn.Sequential(
            nn.Linear(256 * 7 * 7, 512),  # 12544 -> 512
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, 512)  # 512 -> 512 (final cell representation)
        )
        
        # Transformer Encoder for modeling cell relationships
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=512,  # Input dimension matches MLP output
                nhead=8,
                dim_feedforward=2048,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=3
        )
        
        # 2D positional embeddings for cell grid positions
        self.pos_embed = nn.Parameter(torch.randn(num_rows * num_cols, 512))
        
        # Final classification layer for OTSL token prediction
        # OTSL tokens: fcel, ecel, lcel, nl (4 types)
        self.otsl_vocab = {'fcel': 0, 'ecel': 1, 'lcel': 2, 'nl': 3}
        self.classifier = nn.Linear(512, len(self.otsl_vocab))
        
    def forward(self, x, grid_cells):
        """
        Args:
            x: Input image tensor [batch, 3, H, W]
            grid_cells: Grid cell coordinates from Split model
                       List of cell dictionaries with keys: row_start, row_end, col_start, col_end
        """
        # Extract features using modified ResNet-FPN
        fpn_features = self.rfpn(x)  # [batch, 256, H/4, W/4]
        print(f"FPN features shape: {fpn_features.shape}")
        
        # Convert grid cells to ROI format for torchvision ROI Align
        # ROI format: [batch_idx, x1, y1, x2, y2] where coordinates are in original image scale
        rois = []
        batch_size = x.shape[0]
        
        for batch_idx in range(batch_size):
            for row in grid_cells:
                for cell in row:
                    # Convert from grid coordinates to ROI format
                    x1 = cell['col_start']
                    y1 = cell['row_start'] 
                    x2 = cell['col_end']
                    y2 = cell['row_end']
                    
                    # ROI format: [batch_idx, x1, y1, x2, y2]
                    rois.append([batch_idx, x1, y1, x2, y2])
        
        rois = torch.tensor(rois, dtype=torch.float32, device=x.device)
        print(f"ROIs shape: {rois.shape}")
        
        # Apply ROI Align to extract cell features
        cell_features = self.roi_align(
            fpn_features, 
            rois, 
            output_size=self.roi_output_size,
            spatial_scale=self.spatial_scale,
            sampling_ratio=-1  # Adaptive sampling
        )  # [num_rois, 256, 7, 7]
        
        print(f"Cell features shape: {cell_features.shape}")
        
        # Flatten and process through MLP
        flattened_features = self.flatten(cell_features)  # [R*C, 12544]
        cell_embeddings = self.mlp(flattened_features)   # [R*C, 512]
        
        print(f"Cell embeddings shape: {cell_embeddings.shape}")
        
        # Reshape for transformer: [batch, R*C, 512]
        num_cells = len(grid_cells) * len(grid_cells[0]) if grid_cells else 0
        cell_embeddings_batched = cell_embeddings.view(batch_size, num_cells, 512)
        
        # Add positional embeddings
        cell_embeddings_pos = cell_embeddings_batched + self.pos_embed[:num_cells].unsqueeze(0)
        
        # Apply transformer encoder for cell relationship modeling
        transformer_output = self.transformer(cell_embeddings_pos)  # [batch, R*C, 512]
        
        print(f"Transformer output shape: {transformer_output.shape}")
        
        # Final classification for OTSL tokens
        otsl_predictions = self.classifier(transformer_output)  # [batch, R*C, 4]
        
        print(f"OTSL predictions shape: {otsl_predictions.shape}")
        
        return {
            'fpn_features': fpn_features,
            'cell_features': cell_features,      # [R*C, 256, 7, 7] 
            'cell_embeddings': cell_embeddings,  # [R*C, 512]
            'transformer_output': transformer_output,  # [batch, R*C, 512]
            'otsl_predictions': otsl_predictions,     # [batch, R*C, 4]
            'rois': rois
        }
       

def main(image_path):
    """Main function that takes image input and predicts table structure"""
    print("=== TABLET Complete Pipeline ===")
    
    # Step 1: Initialize Split model for row/column detection
    split_model = Split(BasicBlock, [2,2,2,2], fpn_channels=128)
    split_model.eval()
    
    # Step 2: Load and preprocess image
    from PIL import Image
    import torchvision.transforms as transforms
    
    def tablet_transform(image):
        """TABLET preprocessing: resize to 960x960 with padding"""
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
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        return transform(canvas)
    
    # Load image
    img = Image.open(image_path)
    img_tensor = tablet_transform(img).unsqueeze(0)
    
    print(f"Input image shape: {img_tensor.shape}")
    
    # Step 3: Run Split model to detect row/column boundaries
    with torch.no_grad():
        split_outputs = split_model(img_tensor)
        
        # Extract split predictions
        row_splits = split_outputs['row_splits']  # [1, H, 2]
        col_splits = split_outputs['col_splits']  # [1, W, 2]
        
        # Get split probabilities and positions
        row_probs = F.softmax(row_splits, dim=-1)[0, :, 1]
        col_probs = F.softmax(col_splits, dim=-1)[0, :, 1]
        
        row_split_mask = row_probs > 0.5
        col_split_mask = col_probs > 0.5
        
        row_positions = torch.where(row_split_mask)[0].cpu().numpy()
        col_positions = torch.where(col_split_mask)[0].cpu().numpy()
        
        print(f"Detected {len(row_positions)} row splits and {len(col_positions)} column splits")
        
        # Create grid structure
        grid_cells = extract_table_structure(row_positions, col_positions)
        
    # Step 4: Initialize Merge model for cell classification
    num_rows = len(grid_cells)
    num_cols = len(grid_cells[0]) if grid_cells else 0
    merge_model = Merge(BasicBlock, [2,2,2,2], fpn_channels=256, 
                       num_rows=num_rows, num_cols=num_cols)
    merge_model.eval()
    
    print(f"Grid structure: {num_rows} rows × {num_cols} columns")
    
    # Step 5: Run Merge model for cell classification
    with torch.no_grad():
        merge_outputs = merge_model(img_tensor, grid_cells)
        
        otsl_predictions = merge_outputs['otsl_predictions']  # [1, R*C, 4]
        
        # Get predicted OTSL tokens
        predicted_tokens = torch.argmax(otsl_predictions, dim=-1)[0]  # [R*C]
        
        print(f"OTSL predictions shape: {otsl_predictions.shape}")
        print(f"Predicted OTSL tokens: {predicted_tokens}")
        
    # Step 6: Visualize results
    visualize_complete_results(img, row_positions, col_positions, 
                             grid_cells, predicted_tokens, num_rows, num_cols)
    
    return {
        'split_outputs': split_outputs,
        'merge_outputs': merge_outputs,
        'grid_cells': grid_cells,
        'predicted_tokens': predicted_tokens
    }

def visualize_complete_results(original_image, row_positions, col_positions, 
                             grid_cells, predicted_tokens, num_rows, num_cols):
    """Visualize complete TABLET results"""
    import matplotlib.pyplot as plt
    
    # OTSL token mapping
    token_names = ['fcel', 'ecel', 'lcel', 'nl']
    token_colors = ['lightgreen', 'white', 'lightblue', 'lightcoral']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Original image with splits
    axes[0, 0].imshow(original_image)
    img_h, img_w = np.array(original_image).shape[:2]
    
    for pos in row_positions:
        y = int(pos * img_h / 960)
        axes[0, 0].axhline(y=y, color='red', linewidth=2, alpha=0.7)
    
    for pos in col_positions:
        x = int(pos * img_w / 960)
        axes[0, 0].axvline(x=x, color='blue', linewidth=2, alpha=0.7)
    
    axes[0, 0].set_title(f'Detected Grid: {num_rows}×{num_cols}')
    axes[0, 0].axis('off')
    
    # Cell type visualization
    axes[0, 1].imshow(original_image)
    
    for i, row in enumerate(grid_cells):
        for j, cell in enumerate(row):
            cell_idx = i * num_cols + j
            if cell_idx < len(predicted_tokens):
                token_type = predicted_tokens[cell_idx].item()
                
                # Scale coordinates back to original image size
                x1 = int(cell['col_start'] * img_w / 960)
                y1 = int(cell['row_start'] * img_h / 960)
                x2 = int(cell['col_end'] * img_w / 960)
                y2 = int(cell['row_end'] * img_h / 960)
                
                # Draw colored rectangle for OTSL token
                rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                   facecolor=token_colors[token_type], 
                                   alpha=0.6, edgecolor='black')
                axes[0, 1].add_patch(rect)
                
                # Add OTSL token label
                axes[0, 1].text((x1+x2)/2, (y1+y2)/2, 
                              token_names[token_type], 
                              ha='center', va='center', fontsize=8)
    
    axes[0, 1].set_title('OTSL Token Classification')
    axes[0, 1].axis('off')
    
    # OTSL token distribution
    token_counts = torch.bincount(predicted_tokens, minlength=4)
    axes[1, 0].bar(token_names, token_counts.numpy(), color=token_colors)
    axes[1, 0].set_title('OTSL Token Distribution')
    axes[1, 0].set_ylabel('Count')
    
    # Grid structure text summary
    axes[1, 1].text(0.1, 0.8, f'Grid Structure: {num_rows} × {num_cols}', 
                   fontsize=14, weight='bold')
    axes[1, 1].text(0.1, 0.7, f'Total Cells: {num_rows * num_cols}', fontsize=12)
    axes[1, 1].text(0.1, 0.6, 'OTSL Tokens:', fontsize=12, weight='bold')
    
    y_pos = 0.5
    for i, (name, count) in enumerate(zip(token_names, token_counts)):
        axes[1, 1].text(0.1, y_pos, f'{name}: {count.item()}', 
                       fontsize=11, color=cell_colors[i], 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor=token_colors[i], alpha=0.7))
        y_pos -= 0.08
    
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.show()

# Usage
if __name__ == "__main__":
    # Example usage
    image_path = "sample.png"
    results = main(image_path)
