[简体中文](README.md) | English

(Notice: The open-source training code for FourierN1 is still under development and has not been officially released!)

# Wiki-GRx-Gym

This repository provides a training environment based on NVIDIA Isaac Gym, integrating the legged_gym and rsl_rl libraries from ETH Zurich's Legged Robotics team, designed for training GRx robots to
walk on complex terrains.

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
   conda create -n grx-gym python=3.8
   conda activate grx-gym
   ```

3. Dependency Installation
   ```
   # Install Isaac Gym
   cd IsaacGym_Preview_4_Package/isaacgym/python/
   pip install -e . -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

   # Install rsl_rl and legged_gym
   cd project_path/rsl_rl
   pip install -e .
   cd project_path/legged_gym
   pip install -e .

   # Install other dependencies
   pip install numpy==1.20.0 tensorboard protobuf==3.20.3
   ```

### Usage Instructions

1. Start Training
   ```
   cd legged_gym/legged_gym/scripts
   python train.py --task=GRMini1T2 --headless
   ```

2. Run Demo
   ```
   python play.py --task=GRMini1T2 --num_envs=25
   ```

### FAQ

1. Ubuntu 22.04 error "libpython3.8.so.1.0: cannot open shared object file"
    - Solution reference: https://blog.csdn.net/weixin_43989965/article/details/136612205

---

Thank you for your interest in Fourier Intelligence's GRx robot project!  
We hope this resource will provide strong support for your robotics development!