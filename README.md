# FoGRAL

FoGRAL is a graph neural network framework for **drug repositioning (DR)**.  
It aims to improve relation learning on dense drug–drug and disease–disease homogeneous graphs by adaptively suppressing noisy relations and enhancing task-relevant ones.
- **Adaptive homogeneous graph learning**  
  Instead of using static Top-K sparsification, FoGRAL learns task-adaptive masks from drug and disease features to refine dense homogeneous graphs.

- **Noise suppression for dense graphs**  
  Learnable mask matrices are applied to predefined homogeneous graphs via Hadamard product, helping reduce irrelevant noisy relations.

- **Regularized sparse structure learning**  
  L1 regularization and entropy regularization are used to optimize the learned masks and encourage effective sparse graph structures.

- **Multimodal feature alignment**  
  A multimodal alignment loss is introduced to align heterogeneous and homogeneous relational feature spaces for robust feature fusion.

## Requirements

Tested environment:

- Ubuntu 22.04
- Python 3.12.4
- PyTorch 2.3.0 + CUDA 12.1
- PyTorch Geometric 2.7.0

Recommended packages:

```bash
pip install torch torchvision torchaudio
pip install torch-geometric==2.7.0
pip install pandas scipy scikit-learn

