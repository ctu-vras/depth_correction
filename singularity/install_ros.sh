#!/bin/bash

# setup timezone
#echo 'Etc/UTC' > /etc/timezone && \
#    ln -s /usr/share/zoneinfo/Etc/UTC /etc/localtime && \
#    apt-get update && apt-get install -q -y tzdata && rm -rf /var/lib/apt/lists/*


# install packages
apt-get update && apt-get install -q -y \
    dirmngr \
    gnupg2 \
    lsb-release \
    && rm -rf /var/lib/apt/lists/*

# setup keys
apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654

# setup sources.list
echo "deb http://packages.ros.org/ros/ubuntu `lsb_release -sc` main" > /etc/apt/sources.list.d/ros-latest.list

# install bootstrap tools
apt-get update && apt-get install --no-install-recommends -y \
    python3-rosdep \
    python3-rosinstall \
    python3-vcstools \
    && rm -rf /var/lib/apt/lists/*

# setup environment
LANG=C.UTF-8
LC_ALL=C.UTF-8

ROS_DISTRO=melodic
rosdep init && rosdep update --rosdistro $ROS_DISTRO

# install ROS packages
apt-get update && apt-get install -y \
    ros-$ROS_DISTRO-desktop \
    ros-$ROS_DISTRO-ros-numpy \
    && rm -rf /var/lib/apt/lists/*

source /opt/ros/${ROS_DISTRO}/setup.bash
