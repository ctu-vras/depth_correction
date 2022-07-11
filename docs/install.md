# Installation

## Locally

Tested with Ubuntu 20.04 and ROS-noetic.

1. Create ROS workspace and clone the package:
   ```bash
   mkdir -p ~/catkin_ws/src/
   cd ~/catkin_ws/src/
   git clone https://github.com/RuslanAgishev/depth_correction.git
   ```
2. Install python dependencies:
   ```bash
   cd ~/catkin_ws/src/depth_correction/
   pip install -r python_requirements.txt
   pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
   ```
3. Install [ROS](http://wiki.ros.org/ROS/Installation)
4. Install ROS dependencies:
   ```bash
   sudo apt-get install python3-catkin-tools python3-pcl ros-noetic-ros-numpy ros-noetic-rviz ros-noetic-tf-conversions
   ```

6. Build ROS workspace:
   ```bash
   cd ~/catkin_ws/src/ && git clone https://github.com/tpet/data.git
   cd ~/catkin_ws/ && wstool init src ~/catkin_ws/src/depth_correction/dependencies.rosinstall
   cd ~/catkin_ws/ && catkin build
   source devel/setup.bash
   ```

## Singularity image

1. Install [Singularity](https://github.com/RuslanAgishev/depth_correction/blob/main/docs/singularity.md).

2. Building Singularity image.

   ```bash
   cd ../singularity
   sudo singularity build depth_correction.simg depth_correction.txt
   ```

3. Run demo (optionally).

   - Download [ASL laser](https://projects.asl.ethz.ch/datasets/doku.php?id=laserregistration:laserregistration):
   ```bash
   cd ~/catkin_ws/src/data/asl_laser/
   ./get
   ```

   - Ones, you have the singularity image build, it would be possible to run the package inside the environment as follows.
   In the example bellow, first, we bind the up to date package with data to our image.
   Then we source the ROS workspace inside the image.
   The next step is to launch the demo using a sequence from the
   [ASL laser](https://projects.asl.ethz.ch/datasets/doku.php?id=laserregistration:laserregistration).

   ```bash
   singularity shell --nv --bind ~/catkin_ws/src/data/:/opt/ros/depthcorr_ws/src/data/ \
                          --bind ~/catkin_ws/src/depth_correction/:/opt/ros/depthcorr_ws/src/depth_correction/ \
               depth_correction.simg

   source /opt/ros/noetic/setup.bash
   source /opt/ros/depthcorr_ws/devel/setup.bash

   roslaunch depth_correction demo.launch rviz:=True
   ```
