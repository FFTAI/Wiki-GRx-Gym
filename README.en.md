[简体中文](README.md) | English

> [!NOTE]
> The FourierN1 open-source training code is still under development and improvement, so there may be some instability issues during operation!
> If meet issues, feel free to raise an issue 😊

# Wiki-GRx-Gym

This repository provides a training environment based on NVIDIA Isaac Gym, combined with ETH Zurich's Legged Robotics team's legged_gym and rsl_rl libraries, for training Fourier N1 robot's locomotion capabilities on complex terrains.

### Related Resources

* NVIDIA Isaac Gym: https://developer.nvidia.com/isaac-gym
* legged_gym: https://github.com/leggedrobotics/legged_gym.git
* rsl_rl: https://github.com/leggedrobotics/rsl_rl.git

### Installation Guide

1. Install Ubuntu 22.04 system

2. Conda Environment Setup
   ```
   # Install Miniconda
   cd ~/Downloads
   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
   bash Miniconda3-latest-Linux-x86_64.sh

   # Create training environment
   conda create -n wiki-grx-gym python=3.8 -y
   conda activate wiki-grx-gym
   ```

3. Dependency Installation
   ```
   # Navigate to project directory
   cd path/to/your/project
   
   # Install Isaac Gym
   cd IsaacGym_Preview_4_Package/isaacgym/python/
   pip install -e . -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

   # Navigate to project directory
   cd path/to/your/project

   # Install rsl_rl
   cd rsl_rl
   pip install -e . -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

   # Navigate to project directory
   cd path/to/your/project
   
   # Install legged_gym
   cd legged_gym 
   pip install -e . -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

   # Install other dependencies
   pip install tensorboard protobuf==3.20.3
   ```

### Usage Instructions

1. Start Training
   ```
   cd legged_gym/legged_gym/scripts
   python train.py --task=N1 --headless
   ```

2. Run Demo
   ```
   python play.py --task=N1 --num_envs=1
   ```

3. Export Policy:
    - When running `play.py`, the policy network model will be automatically exported to `logs/N1/exported/policy_jit.pt`

### Frequently Asked Questions

1. Ubuntu 22.04 Error "libpython3.8.so.1.0: cannot open shared object file"
    - Error message: ImportError: libpython3.8.so.1.0: cannot open shared object file: No such file or directory
    - Solution reference: https://blog.csdn.net/weixin_43989965/article/details/136612205
    - You can try running the automatic configuration script in this project, then exit the conda environment and reactivate it:
        - `bash shell/conda_import_libpython.sh`
        - `conda deactivate`
        - `conda activate wiki-grx-gym`

---

Thank you for your interest in Fourier Intelligence's N1 robot project!
We hope this resource will provide strong support for your robotics development!
