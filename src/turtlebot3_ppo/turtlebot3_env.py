import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
import numpy as np
import gym  # library for reinforcement learning environments
from gym import spaces  # To define action and observation spaces

class TurtleBot3Env(Node, gym.Env):  # Class that inherits from Node and gym.Env
    def __init__(self):
        super().__init__('turtlebot3_env')  # Initialize the node
        
        # Define action and observation spaces
        self.action_space = spaces.Box(low=np.array([-0.2, -2.8]), high=np.array([0.2, 2.8]), dtype=np.float32)  # Continuous action spaces for linear velocity and angular velocity as defined by the turtlebot sim. (spaces.Box defines this continuous action space)
        # Adjust observation space to include both LIDAR and odometry data
        self.observation_space = spaces.Box(low=0, high=10, shape=(20+5,), dtype=np.float32)  # Setting sensor data range and shape of data (20 LIDAR points + 5 odometry values: x,y,theta,v,omega)
        
        # Create subscriptions and publisher
        self.lidar_sub = self.create_subscription(LaserScan, '/scan', self.lidar_callback, 10)  # Creating a subscriber for LIDAR data
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)  # Creating a subscriber for odometry data
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)  # Creating a publisher for command velocities
        
        self.lidar_data = None  # Store received LIDAR data
        self.odom_data = None  # Store received odometry data
        self.done = False  # Flag for when episode is done

    def lidar_callback(self, msg):  # Convert LiDAR data into numpy array and store it
        self.lidar_data = np.array(msg.ranges)
    
    def odom_callback(self, msg):
        # self.odom_data = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.orientation.z, msg.twist.twist.linear.x, msg.twist.twist.angular.z])
        # Extracting position and velocity from Odometry message
        self.odom_data = {
            'position': {
                'x': msg.pose.pose.position.x,
                'y': msg.pose.pose.position.y,
                'z': msg.pose.pose.position.z
            },
            'orientation': {
                'x': msg.pose.pose.orientation.x,
                'y': msg.pose.pose.orientation.y,
                'z': msg.pose.pose.orientation.z,
                'w': msg.pose.pose.orientation.w
            },
            'linear_velocity': {
                'x': msg.twist.twist.linear.x,
                'y': msg.twist.twist.linear.y,
                'z': msg.twist.twist.linear.z
            },
            'angular_velocity': {
                'x': msg.twist.twist.angular.x,
                'y': msg.twist.twist.angular.y,
                'z': msg.twist.twist.angular.z
            }
        }

    # Initialize the environment by "resetting" it
    def reset(self):
        self.done = False
        rclpy.spin_once(self) # Run this node once to get some data
        if self.lidar_data is None:
            raise RuntimeError("LIDAR data not received yet.")
        return self._get_state()  # Return the combined state (LIDAR + odometry)
    
    def step(self, action):
        cmd_vel = Twist() # define the cmd_vel object from Twist
        cmd_vel.linear.x = float(action[0][0])
        cmd_vel.angular.z = float(action[0][1])
        self.cmd_vel_pub.publish(cmd_vel)  # Send the action to the cmd_vel topic
        
        rclpy.spin_once(self)
        if self.lidar_data is None or self.odom_data is None:
            raise RuntimeError("LIDAR or odometry data not received yet.")
        
        state = self._get_state()  # Get the combined state (LIDAR + odometry)
        reward, self.done = self.get_reward()
        return state, reward, self.done, {}
            
        # Reward function
    def get_reward(self):
        cone_location = np.array([2, -0.5])  # Change according to where the cone is
        goal_dist = np.sqrt((self.odom_data['position']['x'] - cone_location[0]) ** 2 +
                            (self.odom_data['position']['y'] - cone_location[1]) ** 2)  # Euclidean distance between robot and the goal
        
        r_collision = -1/np.mean(self.lidar_data)  # Reward the robot for being far from obstacles
        lr = 20 #learning rate for weighting getting to the goal as more important
        r_goal = lr / (goal_dist + 1e-3)  # Reward the robot for being closer to the goal (avoid division by zero)
        r_non_stationary = (self.odom_data['linear_velocity']['x'] +  
                            self.odom_data['angular_velocity']['z'])  # Reward the robot for moving
        reward = r_collision + r_goal + r_non_stationary
        
        if goal_dist <= .1:
            self.done = True
        
        return reward, self.done

    def clean_data(self):
        # Clean data by removing inf values by replacing them with the max distance number of 10
        inf_indices = np.isinf(self.lidar_data)
        self.lidar_data[inf_indices] = 10
        
        # Sort the distances, get the minimum 20 values
        self.lidar_data = np.sort(self.lidar_data)[:20]

    def _get_state(self):
        self.clean_data()
        # Concatenate LIDAR data with relevant odometry information
        if self.odom_data is None:
            odom = np.array([-2,-.5,0,0,0]) # starting position for standard turtlebot sim is this
        else:
            odom = np.array([self.odom_data['position']['x'], self.odom_data['position']['y'], self.odom_data['orientation']['z'], self.odom_data['linear_velocity']['x'], self.odom_data['angular_velocity']['z']])
        state = np.concatenate([self.lidar_data, odom])  # Combine LIDAR and odometry data into a single state
        return state
    
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
