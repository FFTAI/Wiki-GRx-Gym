# Fourier Intelligence GRx in Isaac Gym Environment

![](./pictures/11.png)

This repository provides an environment used to train GRx to walk on rough terrain using NVIDIA's Isaac Gym, legged_gym and rsl_rl libraries from Legged Robotics @ ETH Zürich.

### Useful Links ###

* NVIDIA Isaac Gym: https://developer.nvidia.com/isaac-gym
* legged_gym: https://github.com/leggedrobotics/legged_gym.git
* rsl_rl: https://github.com/leggedrobotics/rsl_rl.git

### 安装说明 ###

0. 安装**Ubuntu 20.04**，由于 Issac Gym 最高只能在 Ubuntu 20.04 环境下运行，因此只能安装 Ubuntu 20.04 系统。

    * 官方镜像：https://releases.ubuntu.com/focal/
    * 安装教程：https://ubuntu.com/tutorials/install-ubuntu-desktop#1-overview

1. 安装Anaconda: https://www.anaconda.com/download/

2. 安装nvidia驱动
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

3. 创建python虚拟环境，默认环境名为`rlgpu`。
    ```
    cd ./IsaacGym_Preview_4_Package/isaacgym && ./create_conda_env_rlgpu.sh
    ```

4. 验证Isaac Gym 安装：

    - 激活 `conda` 环境;

    ```
    conda activate rlgpu
    ```

    - 运行预装示例;

    ```
    cd examples && python 1080_balls_of_solitude.py
    ```

    - 疑问参考 [isaacgym/docs/index.html](IsaacGym_Preview_4_Package/isaacgym/docs/index.html)

5. 安装 rsl_rl, [rsl_rl/README](rsl-rl/README.md)

    ```
    cd rsl_rl && git checkout v1.0.2 && pip install -e .
    ```

7. 安装 legged_gym, [legged_gym/README](legged-gym/README.md)

    ```
    cd legged_gym && pip install -e .
    ```

8. 其他依赖

    - `pip install numpy==1.20.0`, 部分函数使用旧版变量类型，故 `numpy`版本大于1.24会报错

9. 测试安装；

    ``` 
    cd legged_gym/legged_gym/scripts && python ./train.py --task gr1t1 --num_envs 32
    ```

10. 其他：

    - 目前版本头部及手腕为未激活状态。

