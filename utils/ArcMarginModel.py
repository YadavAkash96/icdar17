import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ArcMarginModel(nn.Module):
    def __init__(self, in_features, out_features, scale=64.0, margin=0.5,sub_centers=3):
        """
        ArcFace Loss (Additive Angular Margin Loss)

        Args:
            in_features (int): Dimension of input features (embedding size).
            out_features (int): Number of classes.
            scale (float): Feature scaling factor (default 64.0).
            margin (float): Additive angular margin (default 0.5).
        """
        super(ArcMarginModel, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = scale
        self.m = margin
        self.sub_centers=sub_centers

        # Initialize weights
        #self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight = nn.Parameter(torch.FloatTensor(sub_centers, out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        # Precompute cos(m) and sin(m) for margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        # Threshold for numerical stability in the cos(theta + m) calculation
        self.th = math.cos(math.pi - margin)
        # -sin(m) * margin scaling factor
        self.mm = math.sin(math.pi - margin) * margin

    
    def forwardOriginalArcFace(self, emb, label):
        """
        Forward pass for ArcFace loss.

        Args:
            input (torch.Tensor): Input features with shape (batch_size, in_features).
            label (torch.Tensor): Ground truth labels with shape (batch_size).

        Returns:
            torch.Tensor: Computed ArcFace loss.
        """
        # Normalize input and weight to unit vectors
        embedding = F.normalize(emb)
        weights = F.normalize(self.weight,dim=2)
        cosine = F.linear(embedding, weights)  # Cosine similarity (cos(theta))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))  # Sine similarity (sin(theta))

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
        loss = F.cross_entropy(output, label)
        return loss
    
    def forward(self, emb, label):
        """
        Forward pass for Sub-center ArcFace loss.
    
        Args:
            emb (torch.Tensor): Input features with shape (batch_size, in_features).
            label (torch.Tensor): Ground truth labels with shape (batch_size).
    
        Returns:
            torch.Tensor: Computed Sub-center ArcFace loss.
        """
        # Normalize input and weight to unit vectors
        embedding = F.normalize(emb)
        weights = F.normalize(self.weight, dim=2)
    
        # Calculate cosine similarity for each sub-center
        cos_theta_all = torch.einsum('bf,kcf->bkc', embedding, weights)  # Shape: (batch_size, sub_centers, out_features)
    
        # For each class, select the maximum similarity across sub-centers
        cos_theta, _ = torch.max(cos_theta_all, dim=1)  # Shape: (batch_size, out_features)
    
        # Sine similarity (sin(theta)) calculation
        sine = torch.sqrt(1.0 - torch.clamp(cos_theta ** 2, 0, 1))
    
        # Calculate cos(theta + margin)
        phi = cos_theta * self.cos_m - sine * self.sin_m
    
        # Apply the condition: only add margin if cosine > threshold
        phi = torch.where(cos_theta > self.th, phi, cos_theta - self.mm)
    
        # One-hot encode the labels
        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
    
        # Adjust the logits by applying the margin only to the correct class logits
        output = (one_hot * phi) + ((1.0 - one_hot) * cos_theta)
    
        # Scale the logits
        output *= self.s
    
        # Compute the loss with cross-entropy
        loss = F.cross_entropy(output, label)
        return loss

if __name__ == "__main__":
    batch_size = 512
    embedding_size = 6400
    num_classes = 5000
    subcenter=3
    # Sample embeddings and labels
    input_embeddings = torch.randn(batch_size, embedding_size)
    labels = torch.randint(0, num_classes, (batch_size,))
    # Initialize ArcFace loss
    metric_arcface = ArcMarginModel(in_features=embedding_size, out_features=num_classes, scale=64.0, margin=0.4,sub_centers=subcenter)
    metric_arcface = nn.DataParallel(metric_arcface)
    metric_arcface = metric_arcface.cuda()
    labels=labels.cuda()
  
  # Calculate loss
    loss = metric_arcface(input_embeddings, labels)

    print("Loss:", loss.item())
    




