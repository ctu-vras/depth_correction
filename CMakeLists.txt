cmake_minimum_required(VERSION 3.0.2)
project(depth_correction)

add_compile_options(-std=c++14)

find_package(catkin REQUIRED)
catkin_python_setup()
catkin_package()

include_directories()

catkin_install_python(PROGRAMS
  scripts/cloud_filter
  scripts/depth_correction
  scripts/robot_data
  scripts/train_demo
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY
    launch/
    DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)
