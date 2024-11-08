# Evaluation of ArcFace with DenseNet and NetRVLAD for Writer Identification

Writer retrieval identifies a document's author within a dataset by finding handwriting similarities [1]. State-of-the-art approaches use SIFT [1, 2, 3] for feature extraction and aggregate encoded embeddings to generate global page descriptors. In historical datasets, unsupervised methods [4] substitute supervised networks [1, 2, 5] due to data degradation and limited samples per writer, as seen in ICDAR17s. We encode the embeddings of our neural network by Random NetVLAD (NetRVLAD) [2], particularly designed for writer retrieval by removing normalization layers and the initialization.

## Method Overview

This project presents an unsupervised approach for writer retrieval using a Convolutional Neural Network (DenseNet) trained on 32x32 patches extracted at SIFT keypoints. Target labels correspond to 5000 classes generated through k-means clustering of descriptors [4]. This work has two main objectives:

1. Reproducing the current state-of-the-art methods.
2. Applying various loss techniques with DenseNet.

## Figure

![image](https://github.com/user-attachments/assets/70e30682-8703-497d-8acd-ffeef4578612) **Fig: Network Architecture**



## Summary of Contributions

- Implementation of DenseNet with Transition Layers
- Hyperparameter tuning experiments to optimize performance
- Implementation and comparison of Additive Margin Softmax Loss, ArcFace, and Triplet Loss
- Additional experiments with data augmentation (morphological operations), variations in feature extraction methods, and different clustering techniques such as DBSCAN and HDBSCAN

## References

1. Peer et al., "Towards Writer Retrieval with SIFT-based Features," 2023.
2. Peer et al., "Random NetVLAD for Writer Retrieval," 2023.
3. Low et al., "Distinctive Image Features from Scale-Invariant Keypoints," 2004.
4. Christlein et al., "Unsupervised Methods for Historical Document Retrieval," 2017.
5. Fiel et al., "Supervised Methods in Writer Retrieval," 2023.

