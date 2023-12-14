# GR1T1 in Isaac Gym Environment #
This repository provides an environment used to train GR1T1 (and other robots) to walk on rough terrain using NVIDIA's Isaac Gym, legged_gym and rsl_rl libraries from Legged Robotics @ ETH Zürich.

### Useful Links ###

* NVIDIA Isaac Gym: https://developer.nvidia.com/isaac-gym
* legged_gym: https://github.com/leggedrobotics/legged_gym.git
* rsl_rl: https://github.com/leggedrobotics/rsl_rl.git

### 安装说明 ###

0. 安装Ubuntu 20.04，由于 Issac Gym 最高只能在 Ubuntu 20.04 环境下运行，因此只能安装 Ubuntu 20.04 系统。
    * 官方镜像：https://releases.ubuntu.com/focal/
    * 安装教程：https://ubuntu.com/tutorials/install-ubuntu-desktop#1-overview
1. 安装Anaconda: https://www.anaconda.com/download/
2. 创建python虚拟环境，将`env_name`替换为自定义的环境名。
    - `conda create -n env_name python=3.8`
3. 安装nvidia驱动
    - 使用 Ubuntu 20.04 自带的软件 Software & Updates 安装 Nvidia 显卡驱动。

    - 需要确保在终端使用命令行 `nvidia-smi` 能看到显卡信息和 CUDA 信息。如图示例所示：

        ```
        +-----------------------------------------------------------------------------+
        | NVIDIA-SMI 525.125.06   Driver Version: 525.125.06   CUDA Version: 12.0     |
        |-------------------------------+----------------------+----------------------+
        | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
        | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
        |                               |                      |               MIG M. |
        |===============================+======================+======================|
        |   0  NVIDIA GeForce ...  Off  | 00000000:01:00.0  On |                  Off |
        |  0%   42C    P8    25W / 450W |    709MiB / 24564MiB |      1%      Default |
        |                               |                      |                  N/A |
        +-------------------------------+----------------------+----------------------+
                                                                                    
        +-----------------------------------------------------------------------------+
        | Processes:                                                                  |
        |  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
        |        ID   ID                                                   Usage      |
        |=============================================================================|
        |    0   N/A  N/A      1032      G   /usr/lib/xorg/Xorg                 53MiB |
        |    0   N/A  N/A      1666      G   /usr/lib/xorg/Xorg                239MiB |
        |    0   N/A  N/A      1805      G   /usr/bin/gnome-shell              125MiB |
        |    0   N/A  N/A      2171      G   /usr/lib/firefox/firefox          205MiB |
        |    0   N/A  N/A      2847      G   ...RendererForSitePerProcess       45MiB |
        |    0   N/A  N/A      3721      G   ...RendererForSitePerProcess       20MiB |
        +-----------------------------------------------------------------------------+
        ```
4. 安装pytorch, https://pytorch.org/get-started/locally/
    - `conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia`
5. 安装 Isaac Gym
    - 从官网(https://developer.nvidia.com/isaac-gym)下载 Isaac Gym, 仓库中已有为preview 4.
    - `cd IsaacGym_Preview_4_Package/isaacgym/python/ && pip install -e .`
    - 试运行程序 `cd examples && python 1080_balls_of_solitude.py` 以确认安装成功
    - 疑问参考 [isaacgym/docs/index.html](IsaacGym_Preview_4_Package/isaacgym/docs/index.html)
6. 安装 rsl_rl, [rsl_rl/README](rsl_rl/README.md)
    - `cd rsl_rl && git checkout v1.0.2 && pip install -e .`
7. 安装 legged_gym, [legged_gym/README](legged_gym/README.md)
    - `cd legged_gym && pip install -e .`
8. 其他依赖
    - `pip install numpy==1.20.0`, 部分函数使用旧版变量类型，故 `numpy`版本大于1.24会报错

### 测试安装 ###

- `conda activate env_name` 将`env_name`替换为自定义的环境名
- `cd legged_gym/legged_gym/scripts/`
- `python train.py --task=gr1t1 -num_envs=512`
如果能出现机器人的训练画面，则环境配置成功，进入了训练过程。

