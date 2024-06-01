#!/bin/bash
set -e
set -u

pip install -e ./IsaacGym_Preview_4_Package/isaacgym/python
pip install -e ./rsl_rl
pip install -e ./legged_gym

/bin/bash