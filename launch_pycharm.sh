#!/bin/bash
# Activate the ROS environment and then launch pycharm

PYCHARM_DIR=pycharm-community-2018
source ../../devel/setup.bash && /opt/${PYCHARM_DIR}/bin/pycharm.sh &