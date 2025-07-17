[English](README.en.md) | 简体中文

# Wiki-GRx-Gym

本仓库提供基于 NVIDIA Isaac Gym 的训练环境，结合苏黎世联邦理工 Legged Robotics 团队的 legged_gym 和 rsl_rl 库，用于训练 Fourier N1 机器人在复杂地形上的行走能力。

### 相关资源

* NVIDIA Isaac Gym: https://developer.nvidia.com/isaac-gym
* legged_gym: https://github.com/leggedrobotics/legged_gym.git
* rsl_rl: https://github.com/leggedrobotics/rsl_rl.git

### 安装指南

1. 安装 Ubuntu 20.04 / Ubuntu 22.04 系统

2. Conda环境配置
   ```
   # 安装Miniconda
   cd ~/Downloads
   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
   bash Miniconda3-latest-Linux-x86_64.sh

   # 创建训练环境
   conda create -n wiki-grx-gym python=3.8 -y
   conda activate wiki-grx-gym
   ```

3. 依赖库安装
   ```
   # 进入到项目目录
   cd path/to/your/project
   
   # 安装Isaac Gym
   cd IsaacGym_Preview_4_Package/isaacgym/python/
   pip install -e . -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

   # 进入到项目目录
   cd path/to/your/project

   # 安装 rsl_rl
   cd rsl_rl
   pip install -e . -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

   # 进入到项目目录
   cd path/to/your/project
   
   # 安装 legged_gym
   cd legged_gym 
   pip install -e . `-i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple`

   # 安装其他依赖
   pip install tensorboard protobuf==3.20.3 -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
   ```

### 使用说明

1. 启动训练
   ```
   cd legged_gym/legged_gym/scripts
   python train.py --task=N1 --headless
   ```

2. 演示测试
   ```
   python play.py --task=N1 --num_envs=1
   ```

3. 导出策略
    - 运行 `play.py` 的时候，会自动导出策略网络模型到 `logs/N1/exported/policy_jit.pt`
    - 该策略模型可用于后续真实机器人的部署。

> [!NOTE]
> 
> 使用 NVIDIA RTX 4090 的平均训练时间，每个 iteration 约为 9.5 秒。
> 完成 5000 个 iteration 需要大约 12 小时。实际训练时间可根据奖赏增长情况和训练目标进行调整。


### 常见问题

1. Ubuntu 22.04 报错 "libpython3.8.so.1.0: 无法打开共享对象文件"
    - 显示信息为：ImportError: libpython3.8.so.1.0: cannot open shared object file: No such file or directory
    - 解决方案参考：https://blog.csdn.net/weixin_43989965/article/details/136612205
    - 可以尝试先激活 `wiki-grx-gym` 环境，运行本项目下自动配置脚本，然后先退出 conda 环境，然后重新激活对应的 conda 环境：

   ```
   conda activate wiki-grx-gym
   bash shell/conda_import_libpython.sh
   conda deactivate
   conda activate wiki-grx-gym
   ```

---

感谢您对傅利叶智能 N1 机器人项目的关注！
希望本资源能为您的机器人开发提供有力支持！