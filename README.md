# "Who is that Pokémon?" with Matching Networks

This project provides an implementation of Matching Networks as described in the paper [Matching Networks for One Shot Learning](http://papers.nips.cc/paper/6385-matching-networks-for-one-shot-learning.pdf) in PyTorch.

This is based in the implementation made on the paper [Fast visual grounding in interaction: bringing few-shot learning with neural networks to an interactive robot](https://aclanthology.org/2020.pam-1.7/), which stems from my Master's Thesis project for the Master's in Language Technology at the University of Gothenburg.

## Minimum requirements
- Python 3
- [NumPy](http://www.numpy.org/)
- [OpenCV-python](https://pypi.org/project/opencv-python/)
- [PyTorch](https://pytorch.org/)

## Installation
1. Install ROS Kinetic, the Freenect drivers and rospy. For the Freenect run:
```
sudo apt-get install freenect ros-kinetic-freenect-stack
```

2. Install all the required Python2.7 libraries according to the instructions on their websites.

3. If the ROS enviroment is not ready, run:
```
source /opt/ros/kinetic/setup.bash
```

4. Create a workspace with catkin and Python2 and source it
```
mkdir -p ~/ROS_WORKSPACE_FOLDER/src
cd ~/ROS_WORKSPACE_FOLDER/
catkin_make --cmake-args -DPYTHON_VERSION=2
. ~/ROS_WORKSPACE_FOLDER/devel/setup.bash
```

5. Create and build a package
```
cd ~/ROS_WORKSPACE_FOLDER/src
catkin_create_pkg NAME_OF_THE_PACKAGE std_msgs rospy
```

6. Rebuild the workspace and source it again
```
cd ~/ROS_WORKSPACE_FOLDER/
catkin_make --cmake-args -DPYTHON_VERSION=2
. ~/ROS_WORKSPACE_FOLDER/devel/setup.bash
```

## Running instructions
For running, we recommend you to use two different terminals.

1. Any of the two terminals: source and run ROS
```
source /opt/ros/kinetic/setup.bash
nohup roslaunch freenect_launch freenect.launch &
```

2. Terminal 1: recognise.py
```
. ~/ROS_WORKSPACE_FOLDER/devel/setup.bash
chmod +x ~/ROS_WORKSPACE_FOLDER/src/NAME_OF_THE_PACKAGE/src/dialogue.py
chmod +x ~/ROS_WORKSPACE_FOLDER/src/NAME_OF_THE_PACKAGE/src/recognise.py
cd ~/ROS_WORKSPACE_FOLDER/src/NAME_OF_THE_PACKAGE/src/
rosrun NAME_OF_THE_PACKAGE recognise.py
```

3. Terminal 2: dialogue.py
```
. ~/ROS_WORKSPACE_FOLDER/devel/setup.bash
cd ~/ROS_WORKSPACE_FOLDER/src/NAME_OF_THE_PACKAGE/src/
rosrun NAME_OF_THE_PACKAGE dialogue.py
```

## SOTA dataset  # TODO CHANGE WITH POKEMON DATASET
SOTA (Small Objects daTAset) is a dataset of 400 images distributed equally into 20 categories. These images portrait a single object of interest which is normally centered in the image.

The images were taken using the same Kinect v1 RGB camera. The images taken are resized to 224x224 pixels since it is the default size that VGG16 and most deep image encoders use by default.

SOTA is available inside this same repository. 

This dataset is licensed under Creative Commons Attribution 4.0 International (CC BY 4.0). Anyone is free to share an adapt this dataset as long as appropriate credit is given to the original author.

<a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by/4.0/88x31.png" /></a><br /><span xmlns:dct="http://purl.org/dc/terms/" href="http://purl.org/dc/dcmitype/StillImage" property="dct:title" rel="dct:type">Small Objects daTAset (SOTA)</span> by <a xmlns:cc="http://creativecommons.org/ns#" href="https://github.com/jcanosan/Interactive-robot-with-neural-networks/tree/master/utils/datasets/sota_dataset" property="cc:attributionName" rel="cc:attributionURL">José Miguel Cano Santín</a> is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International License</a>.
