[English](README.en.md) | 简体中文

(注意：FourierN1 开源训练代码仍处于开发中，尚未正式发布!)

# Wiki-GRx-Gym

本仓库提供基于NVIDIA Isaac Gym的训练环境，结合苏黎世联邦理工Legged Robotics团队的legged_gym和rsl_rl库，用于训练GRx机器人在复杂地形上的行走能力。

### 相关资源

* NVIDIA Isaac Gym: https://developer.nvidia.com/isaac-gym
* legged_gym: https://github.com/leggedrobotics/legged_gym.git
* rsl_rl: https://github.com/leggedrobotics/rsl_rl.git

### 安装指南

1. 安装 Ubuntu 22.04 系统

2. Conda环境配置
   ```
   # 安装Miniconda
   cd ~/Downloads
   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
   bash Miniconda3-latest-Linux-x86_64.sh

   # 创建训练环境
   conda create -n grx-gym python=3.8
   conda activate grx-gym
   ```

3. 依赖安装
   ```
   # 安装Isaac Gym
   cd IsaacGym_Preview_4_Package/isaacgym/python/
   pip install -e . -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

   # 安装 rsl_rl 和 legged_gym
   cd 项目路径/rsl_rl
   pip install -e .
   cd 项目路径/legged_gym 
   pip install -e .

   # 安装其他依赖
   pip install numpy==1.20.0 tensorboard protobuf==3.20.3
   ```

### 使用说明

1. 启动训练
   ```
   cd legged_gym/legged_gym/scripts
   python train.py --task=GRMini1T2 --headless
   ```

2. 演示测试  
   ```
   python play.py --task=GRMini1T2 --num_envs=25
   ```

### 常见问题

1. Ubuntu 22.04报错"libpython3.8.so.1.0: 无法打开共享对象文件"
   - 解决方案参考：https://blog.csdn.net/weixin_43989965/article/details/136612205

---

感谢您对傅利叶智能GRx机器人项目的关注！
希望本资源能为您的机器人开发提供有力支持！