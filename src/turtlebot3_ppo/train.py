"This script runs the training of the PPO policy."
"Initial training parameters such as number of episodes and max time steps within an episode are defined"
"It utilizes other functions to get the actions, states, rewards, etc and store them to input to the training function"

import rclpy
import subprocess
import torch
from stop_robot import *
import pandas as pd  # Add this import
import os
import time
import matplotlib.pyplot as plt
## When using python to run:
from turtlebot3_env import TurtleBot3Env
from ppo_agent import PPOAgent
## when using ROS to run:
# from turtlebot3_ppo.turtlebot3_env import TurtleBot3Env
# from turtlebot3_ppo.ppo_agent import PPOAgent

def reset_gazebo_world():
    # Call the reset world service to reset the Gazebo world
    subprocess.run(['ros2', 'service', 'call', '/reset_world', 'std_srvs/srv/Empty'])

def main(args=None):
    rclpy.init(args=args)
    env = TurtleBot3Env()
    agent = PPOAgent(env)

    n_episodes = 300
    max_timesteps = 10000
    
    # Start up turtlebot sim environment
    # os.environ['TURTLEBOT3_MODEL'] = 'burger'
    # subprocess.Popen(['ros2', 'launch', 'turtlebot3_gazebo', 'turtlebot3_world.launch.py']) # If you need to open the simulation
    rewards = []  
    
    # Setup stop object
    stop = StopRobot()
    stop.stop_robot() # Stop the robot motion by calling stop node
    reset_gazebo_world() # Reset the Gazebo world at the end of the episode

    for episode in range(n_episodes):
        start_time = time.time() # Track training time
        
        state = env.reset() # Initial run of the environment node to get initial states
        memory = []
        total_reward = 0
        for t in range(max_timesteps): # episode lasts for a specific time period
            action, log_prob = agent.select_action(state) # Select best action based on the policy
            next_state, reward, done, _ = env.step(action) # Take the next step according to the chosen action and store the next sensor readings and reward of current action
            memory.append((state, action, reward, done, log_prob)) # Store relevant info
            state = next_state
            total_reward += reward
            if done:
                break
            t += 1
        
        timesteps = t    
        stop.stop_robot() # Stop the robot motion by calling stop node
        reset_gazebo_world() # Reset the Gazebo world at the end of the episode
        
        # Save the reward for plotting
        rewards.append(total_reward)
        
        # Update the policy
        agent.train(memory, timesteps)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f'Episode {episode + 1}/{n_episodes}, Total Reward: {total_reward}')
        print(f"Episode elapsed time: {elapsed_time:.4f} seconds")
    
    # Save the trained model
    torch.save(agent.actor_critic.state_dict(), 'ppo_model4.pth')
    env.close()
    
    # Save rewards to a CSV file
    rewards_df = pd.DataFrame(rewards, columns=["Total Reward"])
    rewards_df.to_csv('rewards4.csv', index=False)
    
    # Plot the rewards from each training episode which should increase over time
    plt.plot(rewards, color = 'blue')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.show()

if __name__ == '__main__':
    main()
