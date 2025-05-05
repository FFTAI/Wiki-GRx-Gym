from setuptools import find_packages
from distutils.core import setup

setup(
    name="legged_gym",
    version="1.0.0",
    author="Jason Chen",
    license="BSD-3-Clause",
    packages=find_packages(),
    author_email="xin.chen@fftai.com",
    description="Isaac Gym environments for Legged Robots",
    install_requires=[
        "isaacgym",
        "rsl-rl",
        "matplotlib",
        "numpy==1.21.1",
        "tensorboard",
    ]
)
