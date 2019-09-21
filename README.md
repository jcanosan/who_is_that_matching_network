# Interactive robot with Neural Networks

This repository provides a Python implementation using Keras and Tensorflow of the [Matching Networks for One Shot Learning](http://papers.nips.cc/paper/6385-matching-networks-for-one-shot-learning.pdf) paper and transfer learning in an interactive robot scenario using a Kinect v1 camera with [ROS Kinetic](http://wiki.ros.org/kinetic).

This project is part of the Master's Thesis project for the MLT programme at the University of Gothenburg.

## Requisites
These are the requisites to replicate this robot using the same hardware (Kinect v1). Potentially, this robot could run with any RGB camera as long as it is supported by the ROS community.

- [ROS Kinetic](http://wiki.ros.org/kinetic)
- [An OS supported by ROS Kinetic](http://wiki.ros.org/kinetic#Platforms)
- [Freenect drivers](https://github.com/OpenKinect/libfreenect)
- [rospy](http://wiki.ros.org/rospy)
- Python 2.7
- [NumPy](http://www.numpy.org/)
- [OpenCV-python](https://pypi.org/project/opencv-python/)
- [Keras](https://keras.io/)
- [Tensorflow](https://www.tensorflow.org/install/)

## Installation
1. Install ROS Kinetic, the Freenect drivers and rospy.
```sudo apt-get install freenect ros-kinetic-freenect-stack```
2. Install all the required Python2.7 libraries according to the instructions on their websites.
3. If the ROS enviroment is not ready, run:
```source /opt/ros/kinetic/setup.bash if the environment was not ready```
4. Create a workspace with catkin and Python2 and source it
```
mkdir -p ~/ROS\_WORKSPACE\_FOLDER/src
cd ~/ROS\_WORKSPACE\_FOLDER/
catkin\_make --cmake-args -DPYTHON\_VERSION=2
. ~/ROS\_WORKSPACE\_FOLDER/devel/setup.bash
```
5. Create and build a package
```
cd ~/ROS\_WORKSPACE\_FOLDER/src
catkin\_create\_pkg NAME\_OF\_THE\_PACKAGE std\_msgs rospy
```
6. Rebuild the workspace and source it again
```
cd ~/ROS\_WORKSPACE\_FOLDER/
catkin\_make --cmake-args -DPYTHON\_VERSION=2
. ~/ROS\_WORKSPACE\_FOLDER/devel/setup.bash
```

## Running instructions
For running, we recommend you to use two different terminals.

1. Source and run ROS
```
source /opt/ros/kinetic/setup.bash
nohup roslaunch freenect_launch freenect.launch &
```

2. Terminal 1: recognise.py
```
. ~/ROS\_WORKSPACE\_FOLDER/devel/setup.bash
chmod +x ~/ROS\_WORKSPACE\_FOLDER/src/NAME\_OF\_THE\_PACKAGE/src/dialogue.py
chmod +x ~/ROS\_WORKSPACE\_FOLDER/src/NAME\_OF\_THE\_PACKAGE/src/recognise.py
cd ~/ROS\_WORKSPACE\_FOLDER/src/NAME\_OF\_THE\_PACKAGE/src/
rosrun NAME\_OF\_THE\_PACKAGE recognise.py
```

3. Terminal 2: dialogue.py
```
. ~/ROS\_WORKSPACE\_FOLDER/devel/setup.bash
cd ~/ROS\_WORKSPACE\_FOLDER/src/mlt_project/src/
rosrun NAME\_OF\_THE\_PACKAGE dialogue.py
```