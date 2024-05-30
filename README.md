# Mesh Generator
EAI - Sapienza (AIRO) 2023/2024 -  Bruno Francesco Nocera, Leonardo Colosi


## Table of content
- [Introduction](#introduction)
- [Setup](#set-up-local-environment)
- [Resources](#useful-resources)
<!-- - [References](#references) -->
--- 
# Introduction

**Mesh Generation with Differentiable Rendering and Adversarial Training**

The purpose of this project is to learn a realistic textured mesh using a differentiable rasterizer and adversarial training. Instead of directly optimizing the scene parameters as we saw in class you have to implement a neural network that learns the displacement vectors and a per vertex texture.


# Set up local environment

### conda 
```code 

# python v.3.8
conda create -n mesh_gen_3.8 python==3.8
conda activate mesh_gen 

# or python v.3.9
conda create -n mesh_gen_3.9 python=3.9
conda activate mesh_gen_3.9
```

### fvcode
```code
pip install -U fvcore
```

### pytorch (GPU)
```code
conda install pytorch=1.13.0 torchvision pytorch-cuda=11.6 -c pytorch -c nvidia
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
```

### nvidia
```code
conda install -c bottler nvidiacub
```

### pytorch3d
Note: ***Run only this in colab or in a jupiter notebook***
```code 
import sys
import torch
pyt_version_str=torch.__version__.split("+")[0].replace(".", "")
version_str="".join([
    f"py3{sys.version_info.minor}_cu",
    torch.version.cuda.replace(".",""),
    f"_pyt{pyt_version_str}"
])
%pip install fvcore iopath
%pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/{version_str}/download.html
```

# Useful Resources
To explore some resources used/related to the project look here [resources](resources)

colab/
    MeshGen.py (big notebook ?)
data/
    custom_shapenet/
local/
    dataset/
        setup_dataset.py -> torch dataset + save
    utility/
        utils.py ->   load_target_mesh
                        multiview rendering
                        setup render
                        visualize_collect_data
                        visualize_prediction
                        plot_3d_mesh
                        collect data
                        dict_path
    models/
        baseline.py
        single_mesh_gen.py      --> train on a single mesh at the time
        multiple_mesh_gen.py    --> train on bathes of multiple meshes of the same class


