ABS_PATH="/home/workspace/CarND-Capstone"
#cd $ABS_PATH
#pip install -r $ABS_PATH/requirements.txt
cd $ABS_PATH/ros
catkin_make
source $ABS_PATH/ros/devel/setup.sh
roslaunch $ABS_PATH/ros/launch/styx.launch
