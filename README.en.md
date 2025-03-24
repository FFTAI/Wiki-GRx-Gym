

# Wiki-GRx-Gym

<img src="./pictures/gr1t2_gym.png" width="300" height="360" />


This repository provides an environment used to train GRx to walk on rough terrain using NVIDIA's Isaac Gym,
legged_gym and rsl_rl libraries from Legged Robotics @ ETH Zürich.

### Useful Links

* NVIDIA Isaac Gym: https://developer.nvidia.com/isaac-gym
* legged_gym: https://github.com/leggedrobotics/legged_gym.git
* rsl_rl: https://github.com/leggedrobotics/rsl_rl.git

### Installation

0. Install Ubuntu 20.04 / 22.04.

1. Install Nvidia Driver:
    - Install Nvidia driver using the Software & Updates application that comes with Ubuntu 20.04 / 22.04.
    - Make sure you can see the GPU information and CUDA information by using the command line `nvidia-smi` in the terminal.

2. Conda Environment Setup:

    1. Install Miniconda:
        ```
        cd $HOME/Downloads
    
        wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
        bash Miniconda3-latest-Linux-x86_64.sh
        ```

    2. Create conda environment `grx-gym`:
       ```
       conda create -n grx-gym python=3.8
       conda activate grx-gym
       ```

    3. Install conda environment dependencies:
       ```
       # Install isaacgym
       cd path/to/your/workspace
       
       cd ./IsaacGym_Preview_4_Package/isaacgym/python/
       pip install -e . -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
       
       # Install rsl_rl
       cd path/to/your/workspace
       
       cd ./rsl_rl
       pip install -e . -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
       
       # Install legged_gym
       cd path/to/your/workspace
       
       cd ./legged_gym
       pip install -e . -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
       ```

    4. Install other dependencies:
       ```
       # Some functions use old variable types, so numpy version greater than 1.24 will report an error
       pip install numpy==1.20.0
         
       # tensorboard is needed for display the training process
       pip install tensorboard
       pip install protobuf==3.20.3
       ```

3. Training:

```
cd legged_gym/legged_gym/scripts
python ./train.py --task=GRMini1T2 --headless
```

4. Playing:

```
cd legged_gym/legged_gym/scripts
python ./play.py --task=GRMini1T2 --num_envs=25
```

### Common Issues

1. Ubuntu 22.04 libpython3.8.so.1.0: cannot open shared object file: No such file or directory
    - Solution: https://blog.csdn.net/weixin_43989965/article/details/136612205

---

Thank you for your interest in the Fourier Intelligence GRx Robot Model Repository.
We hope you find this resource helpful in your robotics projects!