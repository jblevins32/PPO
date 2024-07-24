import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
import numpy as np
import gym # library for reinforcement learning environments
from gym import spaces # To define action and observation spaces

class TurtleBot3Env(Node, gym.Env): # Class that inherits from Node and gym.Env
    def __init__(self):
        super().__init__('turtlebot3_env') # initialize the node
        
        self.action_space = spaces.Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32) # continuous action spaces for linear velocity and angular velocity from -1 to 1. spaces.Box defines this continuous action space
        self.observation_space = spaces.Box(low=0, high=10, shape=(360,), dtype=np.float32) # Setting sensor data range and shape of data (360 inputs)
        
        # Creating subscribers and publishers to the LiDAR data and command velocities respectively
        self.lidar_sub = self.create_subscription(LaserScan, '/scan', self.lidar_callback, 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        self.lidar_data = None # Store received LiDAR data
        self.done = False # Flag for when episode is done

    def lidar_callback(self, msg): # Convert LiDAR data into numpy array and store it
        self.lidar_data = np.array(msg.ranges)
    
    # Reset environment each episode
    def reset(self):
        self.done = False
        rclpy.spin_once(self)
        if self.lidar_data is None:
            raise RuntimeError("LIDAR data not received yet.")
        return self.lidar_data
    
    # Take the next action
    def step(self, action):
        cmd_vel = Twist()
        cmd_vel.linear.x = action[0]
        cmd_vel.angular.z = action[1]
        self.cmd_vel_pub.publish(cmd_vel) # send the action to the cmd_vel topic
        
        rclpy.spin_once(self)
        if self.lidar_data is None:
            raise RuntimeError("LIDAR data not received yet.")
        state = self.lidar_data 
        reward = -np.min(state) # Penalize the agent for getting close to obstacles. Need to modify to reward when it is close to the cone.
        
        if self.done: # finishing without reaching the cone
            reward = -100
        
        return state, reward, self.done, {}

    # Clean up when done
    def close(self):
        self.destroy_subscription(self.lidar_sub)
        self.destroy_publisher(self.cmd_vel_pub)
        rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    env = TurtleBot3Env()
    try:
        rclpy.spin(env)
    except KeyboardInterrupt:
        pass
    finally:
        env.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
