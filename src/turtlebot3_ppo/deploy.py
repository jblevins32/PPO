import rclpy
from turtlebot3_env import TurtleBot3Env
from ppo_agent import PPOAgent
import torch
from train import reset_gazebo_world
import time
from stop_robot import *
import rclpy

def main(args=None):
    rclpy.init(args=args)

    # Stop the robot by running the stop_robot script
    stop = StopRobot()
    stop.stop_robot()
    
    reset_gazebo_world() # Reset the Gazebo world at the end of the episode
    env = TurtleBot3Env()
    agent = PPOAgent(env)
    agent.actor_critic.load_state_dict(torch.load('ppo_model.pth'))

    state = env.reset()
    time.sleep(2) # wait for the simulation to be ready
    
    while True:
        action, _ = agent.select_action(state)
        state, _, done, _ = env.step(action)
        if done:
            break
    env.close()
    rclpy.shutdown()
    
    # Stop the robot by running the stop_robot script
    stop.stop_robot()

if __name__ == '__main__':
    main()
