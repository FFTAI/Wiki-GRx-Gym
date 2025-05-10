from setuptools import find_packages
from distutils.core import setup

setup(
    name="legged_gym",
    version="1.0.0",
    description="Isaac Gym environments for Fourier Robots",
    author="Jason Chen",
    author_email="xin.chen@fftai.com",
    license="LGPL-3.0",
    packages=find_packages(),
    install_requires=[
        "isaacgym",
        "rsl-rl",
        "matplotlib",
        "numpy==1.21.1",
        "tensorboard",
    ]
)
