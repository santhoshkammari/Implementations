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

class Merge(nn.Module):
    """TABLET Merge Model following Section 3.2"""
    def __init__(self, fpn_channels=256):
        super().__init__()
        
        # Standard ResNet-18 + FPN (unlike split model, this uses standard backbone)
        # Output: F1/4 of size H/4 × W/4 × 256
        self.rfpn = ResNetFPN(BasicBlock, [2,2,2,2], fpn_channels)
        
        # RoIAlign for extracting 7×7 features from each grid cell
        self.roi_align = nn.AdaptiveAvgPool2d((7, 7))  # or use torchvision.ops.RoIAlign
        
        # Two-layer MLP for dimensionality reduction
        # Input: 7×7×256 = 12544, Output: 512
        self.mlp = nn.Sequential(
            nn.Linear(7 * 7 * fpn_channels, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512)
        )
        
        # Transformer encoder for modeling grid cell relationships
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=512,
                nhead=8,
                dim_feedforward=2048,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=3
        )
        
        # 2D positional embedding for grid cells (learnable)
        # Max grid size assumption: 40×40 = 1600 cells
        self.max_grid_size = 40
        self.pos_embed_2d = nn.Parameter(torch.randn(self.max_grid_size, self.max_grid_size, 512))
        
        # OTSL classification head
        # 5 classes: "C", "L", "U", "X" (no "NL" needed for split-merge)
        self.otsl_classifier = nn.Linear(512, 4)  # C, L, U, X
        
        # Focal loss
        self.focal_loss = FocalLoss(alpha=1.0, gamma=2.0)
        
    def forward(self, x, grid_coords):
        """
        Args:
            x: Input table image [batch, 3, H, W]
            grid_coords: Grid coordinates from split model [R, C, 4] 
                        where each grid cell has [x1, y1, x2, y2] coordinates
        """
        batch = x.size(0)
        
        # Extract FPN features: F1/4 of size H/4 × W/4 × 256
        fpn_features = self.rfpn(x)  # [batch, 256, H/4, W/4]
        
        R, C = grid_coords.shape[:2]  # Grid dimensions
        
        # Extract RoI features for each grid cell
        grid_features = []
        
        for r in range(R):
            for c in range(C):
                # Get coordinates for this grid cell
                x1, y1, x2, y2 = grid_coords[r, c]
                
                # Scale coordinates to match F1/4 feature map
                # If input is H×W, feature map is H/4×W/4
                scale_h, scale_w = fpn_features.size(2) / x.size(2), fpn_features.size(3) / x.size(3)
                x1, y1, x2, y2 = x1 * scale_w, y1 * scale_h, x2 * scale_w, y2 * scale_h
                
                # Extract 7×7 RoI features for this grid cell
                # Simple crop and resize (can replace with proper RoIAlign)
                roi_features = fpn_features[:, :, int(y1):int(y2), int(x1):int(x2)]
                roi_features = F.adaptive_avg_pool2d(roi_features, (7, 7))  # [batch, 256, 7, 7]
                
                grid_features.append(roi_features)
        
        # Stack all grid features: [batch, R×C, 256, 7, 7]
        Fgrids = torch.stack(grid_features, dim=1)  # [batch, R×C, 256, 7, 7]
        
        # Flatten and pass through MLP
        batch_size, num_cells = Fgrids.shape[:2]
        Fgrids_flat = Fgrids.view(batch_size, num_cells, -1)  # [batch, R×C, 7×7×256]
        Sgrids = self.mlp(Fgrids_flat)  # [batch, R×C, 512]
        
        # Add 2D positional embeddings
        # Create position indices for each grid cell
        pos_embeddings = []
        for r in range(R):
            for c in range(C):
                pos_embeddings.append(self.pos_embed_2d[r, c])  # [512]
        pos_embeddings = torch.stack(pos_embeddings).unsqueeze(0)  # [1, R×C, 512]
        
        # Add positional embeddings to features
        Sgrids = Sgrids + pos_embeddings
        
        # Apply transformer encoder to model grid cell relationships
        transformed_features = self.transformer(Sgrids)  # [batch, R×C, 512]
        
        # OTSL classification for each grid cell
        otsl_predictions = self.otsl_classifier(transformed_features)  # [batch, R×C, 4]
        
        # Reshape back to grid format
        otsl_grid = otsl_predictions.view(batch, R, C, 4)  # [batch, R, C, 4]
        
        return {
            'fpn_features': fpn_features,       # [batch, 256, H/4, W/4]
            'grid_features': transformed_features,  # [batch, R×C, 512]
            'otsl_predictions': otsl_grid,      # [batch, R, C, 4] - logits for C/L/U/X
            'grid_shape': (R, C)
        }
        
    def convert_otsl_to_html(self, otsl_predictions, grid_shape):
        """Convert OTSL predictions to HTML table structure"""
        R, C = grid_shape
        otsl_classes = ['C', 'L', 'U', 'X']  # 0=C, 1=L, 2=U, 3=X
        
        # Get predicted classes
        predicted_classes = torch.argmax(otsl_predictions, dim=-1)  # [batch, R, C]
        
        # Convert to OTSL tokens (simplified - full implementation would be more complex)
        html_tables = []
        for b in range(predicted_classes.size(0)):
            otsl_tokens = []
            for r in range(R):
                row_tokens = []
                for c in range(C):
                    class_idx = predicted_classes[b, r, c].item()
                    row_tokens.append(otsl_classes[class_idx])
                otsl_tokens.append(row_tokens)
            html_tables.append(otsl_tokens)
        
        return html_tables

# Usage
model = Split(BasicBlock, [2,2,2,2], fpn_channels=128)
outputs = inference(model)