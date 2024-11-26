# Proximal Policy Optimization with ROS2 and Turtlebot
This ROS2 workspace is for running PPO with turtlebot3 currently concurrent with simulations in Gazebo.

# Getting Started
1) Run `colcon build` in top folder
2) `source install/setup.bash`
3) `export TURTLEBOT3_MODEL=burger`
4) `ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py`
5) run `python3 train.py` to train the agent or `python3 deploy.py` to deploy the agent
