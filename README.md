# FoGRAL

- FoGRAL is a graph neural network framework for drug repositioning (DR).

- It aims to improve relation learning on dense drug–drug and disease–disease homogeneous graphs by adaptively suppressing noisy relations and enhancing task-relevant ones.

- The framework of FoGRAL is as follows:

<br>
<div align=left> <img src="pic/FoGRAL.svg" height="100%" width="100%"/> </div>


## Install based on Ubuntu 22.04

- **Ensure you have installed CUDA 12.1 before installing other packages**

**1.Python environment:** recommending using Conda package manager to install

```python
conda create -n fogral python=3.12.4
conda activate fogral
```

**2.Python package:**
```python
Python == 3.12.4
PyTorch == 2.3.0
CUDA == 12.1
torch-geometric == 2.7.0
scikit-learn == 1.7.2
```

## Run the Experiment
```python
python main.py
```

