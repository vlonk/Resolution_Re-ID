import torch
import torch.nn as nn
import timm

class ReIDResNet50(nn.Module):
    """
    ResNet-50 backbone with a trainable embedding head and optional ID-classification head.
    """
    def __init__(self, num_ids, emb_dim=512, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            'resnet50',
            pretrained=pretrained,
            num_classes=0,
            global_pool='avg'
        )
        self.embedding = nn.Sequential(
            nn.Linear(self.backbone.num_features, emb_dim),
            nn.BatchNorm1d(emb_dim),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(emb_dim, num_ids)

    def forward(self, x):
        """
        x: [B,3,H,W] input images
        returns:
          emb: [B,emb_dim] normalized embedding vectors
          logits: [B,num_ids] raw scores for ID classification
        """
        feats = self.backbone(x)      # → [B, 2048]
        emb   = self.embedding(feats) # → [B, emb_dim]
        logits = self.classifier(emb) # → [B, num_ids]
        return emb, logits
