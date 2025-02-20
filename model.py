import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig
from timm import create_model


class DualStreamFusionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DualStreamFusionBlock, self).__init__()
        self.initial_conv = nn.Sequential(
            nn.Conv2d(2 * in_channels, in_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=1)
        )
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 16, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.final_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, F_C, F_V):
        # Concatenate ConvNeXt and Vision Transformer outputs
        x = torch.cat((F_C, F_V), dim=1)
        x = self.initial_conv(x)

        # Apply channel attention
        attention = self.channel_attention(x)
        x = x * attention

        # Final convolution
        x = self.final_conv(x)
        return x
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.global_avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y
class DualStreamFusionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DualStreamFusionBlock, self).__init__()

        # Initial 1x1 convolutions after concatenation
        self.initial_conv = nn.Sequential(
            nn.Conv2d(2 * in_channels, in_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=1)
        )
        self.conv1_1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv1_2 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv1_3 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv1_4 = nn.Conv2d(3*in_channels, in_channels, kernel_size=1)

        
        self.conv3_1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        
        # Element-wise multiplication and final channel attention
        self.channel_attention = ChannelAttention(in_channels)
        self.final_conv = nn.Conv2d(3 * in_channels, out_channels, kernel_size=1)
        self.norm = nn.LayerNorm(768, eps=1e-6)
    def forward(self, F_C, F_V):
        # Concatenate inputs
        x = torch.cat((F_C, F_V), dim=1)

        # Initial convolution
        x = self.initial_conv(x)

        x_1 = self.conv1_1(x)
        x_1_1 = self.conv1_2(x_1)
        x_1_1 = F.sigmoid(x_1_1)
        
        x_2 = self.conv3_1(x)
        x_2 = F.relu(x_2)
        x_2_1 = self.conv1_3(x_2)
        x_2_1 = F.sigmoid(x_2_1)
        
        x_3 = self.conv3_2(x)
        x_3 = F.relu(x_3)
        x_3 = self.conv3_3(x_3)

        x_2 = x_1_1 * x_2
        x_3 = x_3 * x_2_1
        # Channel attention
        combined = torch.cat((x_1, x_2, x_3), dim=1)
        combined = self.conv1_4(combined)
        attention = self.channel_attention(combined)
        top = x_1 * attention
        middle = x_2 * attention
        bottom = x_3 * attention
        final = torch.cat((top, middle, bottom), dim=1)

        # Final 1x1 convolution
        F_fusion = self.final_conv(final)
        return self.norm(F_fusion.mean([-2, -1]))
class HybridClassifier(nn.Module):
    def __init__(self, num_classes=1000):
        super(HybridClassifier, self).__init__()
        # Load pretrained ConvNeXt V2
        self.convnext = create_model('convnextv2_tiny', pretrained=True, features_only=True)

        # Load pretrained Vision Transformer
        self.vit_config = ViTConfig.from_pretrained("google/vit-base-patch16-224-in21k")
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k", config=self.vit_config)

        # Fusion module
        self.fusion = DualStreamFusionBlock(768, 768)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(768, num_classes)
        )

    def forward(self, x):
        # ConvNeXt feature extraction
        conv_features = self.convnext(x)[-1]  # Shape: (B, C, H, W)

        # Vision Transformer feature extraction
        vit_features = self.vit(pixel_values=x).last_hidden_state  # Shape: (B, N, D)
        B, N, D = vit_features.shape

        # Remove the [CLS] token and reshape
        vit_features = vit_features[:, 1:, :]  # Shape: (B, N-1, D)
        H, W = conv_features.shape[2], conv_features.shape[3]  # Match ConvNeXt spatial size

        # Reshape ViT output to match ConvNeXt spatial dimensions
        vit_features = vit_features.permute(0, 2, 1)  # Shape: (B, D, N-1)
        vit_features = vit_features.reshape(B, D, int((N-1)**0.5), int((N-1)**0.5))  # Shape: (B, D, H', W')

        # Resize ViT features to match ConvNeXt spatial dimensions
        vit_features = nn.functional.interpolate(vit_features, size=(H, W), mode='bilinear', align_corners=False)

        # Fusion module
        fused_features = self.fusion(conv_features, vit_features)

        # Classification head
        logits = self.classifier(fused_features)
        return logits