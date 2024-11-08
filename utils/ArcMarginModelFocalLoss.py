import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ArcMarginModelFocalLoss(nn.Module):
    def __init__(self, in_features, out_features, scale=64.0, margin=0.5, gamma=2.0, alpha=0.25):
        """
        ArcFace Loss (Additive Angular Margin Loss) with Focal Loss.

        Args:
            in_features (int): Dimension of input features (embedding size).
            out_features (int): Number of classes.
            scale (float): Feature scaling factor (default 64.0).
            margin (float): Additive angular margin (default 0.5).
            gamma (float): Focusing parameter for Focal Loss.
            alpha (float): Balancing factor for Focal Loss.
        """
        super(ArcMarginModelFocalLoss, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = scale
        self.m = margin
        self.gamma = gamma
        self.alpha = alpha

        # Initialize weights
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        # Precompute cos(m) and sin(m) for margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        # Threshold for numerical stability in the cos(theta + m) calculation
        self.th = math.cos(math.pi - margin)
        # -sin(m) * margin scaling factor
        self.mm = math.sin(math.pi - margin) * margin

    def focal_loss(self, logits, labels):
        """
        Focal Loss for multi-class classification.

        Args:
            logits (torch.Tensor): Logits from the ArcFace layer.
            labels (torch.Tensor): Ground truth labels with shape (batch_size).

        Returns:
            torch.Tensor: Computed Focal Loss.
        """
        # Convert labels to one-hot encoding
        #one_hot = torch.zeros_like(logits)
        #one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        one_hot = F.one_hot(labels, num_classes=logits.size(-1)).float()
        # Compute the log probabilities
        log_probs = F.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)

        # Apply Focal Loss formula
        focal_weight = torch.pow(1 - probs, self.gamma)
        focal_loss = -self.alpha * focal_weight * log_probs * one_hot
        return focal_loss.sum(dim=1).mean()

    def forward(self, emb, label):
        """
        Forward pass for ArcFace with Focal Loss.

        Args:
            emb (torch.Tensor): Input features with shape (batch_size, in_features).
            label (torch.Tensor): Ground truth labels with shape (batch_size).

        Returns:
            torch.Tensor: Computed ArcFace with Focal Loss.
        """
        # Normalize input and weight to unit vectors
        embedding = F.normalize(emb)
        weights = F.normalize(self.weight)
        cosine = F.linear(embedding, weights)  # Cosine similarity (cos(theta))
        #sine = torch.sqrt(1.0 - torch.pow(cosine, 2))  # Sine similarity (sin(theta))
        sine = torch.sqrt(1.0 - torch.clamp(torch.pow(cosine, 2), min=1e-6))

        # Calculate cos(theta + margin)
        phi = cosine * self.cos_m - sine * self.sin_m

        # Apply condition: only add margin if cosine > threshold
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # Convert the ground truth label to one-hot encoding
        one_hot = torch.zeros_like(cosine)      
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        # Add margin only for ground truth classes
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)

        # Scale the features
        output *= self.s

        # Calculate Focal Loss
        loss = self.focal_loss(output, label)
        return loss

if __name__ == "__main__":
    batch_size = 8
    embedding_size = 6400
    num_classes = 5000
    # Sample embeddings and labels
    input_embeddings = torch.randn(batch_size, embedding_size)
    labels = torch.randint(0, num_classes, (batch_size,))
    # Initialize ArcFace with Focal Loss
    metric_arcface = ArcMarginModelFocalLoss(in_features=embedding_size, out_features=num_classes, scale=64.0, 
                                             margin=0.2, gamma=1.5, alpha=0.5)
    metric_arcface = nn.DataParallel(metric_arcface)
    metric_arcface = metric_arcface.cuda()
    labels = labels.cuda()

    # Calculate loss
    loss = metric_arcface(input_embeddings, labels)

    print("Loss:", loss.item())
