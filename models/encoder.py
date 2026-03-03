import timm
import torch
import torch.nn as nn


class ViTEncoder(nn.Module):
    """Timm-based encoder for vision models (ViT/Swin/ResNet/ConvNeXt).
    
    Input: (B, 3, H, W), Output: (B, D) feature vector per image.
    """

    def __init__(self, model_size: str = 'small'):
        """Initialize ViT encoder with specified size.
        
        Args:
            model_size: 'tiny' or 'small' (default)
        Raises:
            ValueError: Unsupported model size
        """
        super().__init__()

        if model_size == 'tiny':  
            self.backbone = timm.create_model('vit_tiny_patch16_224', pretrained=True)
        elif model_size == 'small':
            self.backbone = timm.create_model('vit_small_patch16_224', pretrained=True)
        else:
            raise ValueError(f"Unsupported model size: {model_size}. Only 'tiny'/'small' are valid.")
        
        # Remove classification head to get raw embeddings
        self.backbone.head = torch.nn.Identity()  

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass to extract image feature vectors.
        
        Args:
            x: Input tensor (B, 3, H, W)
        Returns:
            Feature tensor (B, D)
        """
        feature_vector = self.backbone(x)
        return feature_vector