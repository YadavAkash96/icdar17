# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 21:57:02 2024

@author: Akash
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class AdMSoftmaxLoss(nn.Module):
    """
    Additive Margin Softmax Loss (AM-Softmax Loss) implementation.
    
    This loss function is used for face recognition, speaker recognition, 
    and other tasks where discriminative feature learning is required. It 
    imposes a margin between the target class and non-target classes to 
    improve inter-class separability.
    
    Args:
        embeddings (int): The dimensionality of input feature vectors.
        num_classes (int): The number of output classes.
        s (float): Scaling factor for logits. Default is 30.0.
        m (float): Additive margin. Default is 0.4.
    """
    
    def __init__(self, embeddings, num_classes, s=30.0, m=0.4):
        super(AdMSoftmaxLoss, self).__init__()
        self.s = s  # Scaling factor
        self.m = m  # Margin
        self.embeddings = embeddings  # Dimensionality of input features
        self.num_classes = num_classes  # Number of output classes
        
        # Fully connected layer without bias to learn class weights
        self.weights = nn.Linear(embeddings, num_classes, bias=False).cuda()
    
    def forward(self, x, labels):
        """
        Forward pass of the loss function.
        
        Args:
            x (Tensor): Input features of shape (N, embeddings), where N is the batch size.
            labels (Tensor): Ground truth labels of shape (N,), where N is the batch size.
        
        Returns:
            loss (Tensor): The computed Additive Margin Softmax Loss.
        """
        
        # Ensure the inputs are valid
        assert len(x) == len(labels), "Input size and label size must match"
        assert torch.min(labels) >= 0, "Labels should be non-negative"
        assert torch.max(labels) < self.num_classes, "Labels should be less than number of classes"
        
        # Normalize the class weights (W) to have unit length (L2 normalization)
        for W in self.weights.parameters():
            W = F.normalize(W, dim=1)  # Normalize along the feature dimension
        
        # Normalize input features (L2 normalization)
        x = F.normalize(x, dim=1)
        
        # Compute the weighted cosine similarities (W * x^T)
        wf = self.weights(x)  # Shape: (N, num_classes)
        
        # Extract the logits for the correct classes using ground truth labels
        numerator = self.s * (torch.diagonal(wf.transpose(0, 1)[labels]) - self.m)
        # Numerator: Apply scaling (s) and subtract the margin (m) from the target class logit
        
        # Compute the logits for the non-target classes
        # Exclude the target class logits from each row and keep non-target class logits
        excl = torch.cat([torch.cat((wf[i, :y], wf[i, y+1:])).unsqueeze(0) 
                          for i, y in enumerate(labels)], dim=0)
        
        # Compute the denominator by summing the exponentials of the scaled logits
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * excl), dim=1)
        
        # Calculate the additive margin softmax loss
        L = numerator - torch.log(denominator)
        loss = -torch.mean(L)  # Negative mean of the log likelihood
        
        return loss


if __name__ == "__main__":
    # Example initialization
    num_classes = 5000  # Number of unique authors
    embedding_dim = 6400  # NetRVLAD embedding dimension
    #num_clusters = 64  # Example value for clusters
    
    
    # Example usage
    am_softmax_loss = AdMSoftmaxLoss(embeddings=embedding_dim, num_classes=num_classes, s=60.0, m=0.4)
    embeddings = torch.randn(16, embedding_dim).cuda()  # 16 samples, 512-dimensional embeddings
    labels = torch.randint(0, 10, (16,)).cuda()  # Random labels for 10 classes
    loss = am_softmax_loss(embeddings, labels)
    print("AM-Softmax Loss:", loss)