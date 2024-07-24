import rclpy
from rclpy.node import Node
import numpy as np
import torch
## When using python to run:
from turtlebot3_env import TurtleBot3Env
from ppo_agent import PPOAgent
## when using ROS to run:
# from turtlebot3_ppo.turtlebot3_env import TurtleBot3Env
# from turtlebot3_ppo.ppo_agent import PPOAgent

def main(args=None):
    rclpy.init(args=args)
    env = TurtleBot3Env()
    agent = PPOAgent(env)

    n_episodes = 1000
    max_timesteps = 300

    for episode in range(n_episodes):
        state = env.reset()
        memory = []
        total_reward = 0
        for t in range(max_timesteps):
            action = agent.select_action(state) # Select best action based on the policy
            next_state, reward, done, _ = env.step(action) # Take the next step according to the chosen action
            memory.append((state, action, reward, done)) # Store relevent info
            state = next_state
            total_reward += reward
            if done:
                break
        agent.train(memory) # update the policy
        print(f'Episode {episode + 1}/{n_episodes}, Total Reward: {total_reward}')
        # Need to reset sim environment here
    
    # Save the trained model
    torch.save(agent.actor_critic.state_dict(), 'ppo_model.pth')
    env.close()

if __name__ == '__main__':
    main()
