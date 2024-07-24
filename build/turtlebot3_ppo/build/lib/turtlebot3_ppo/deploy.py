import rclpy
from turtlebot3_env import TurtleBot3Env
from ppo_agent import PPOAgent
import torch

def main(args=None):
    rclpy.init(args=args)
    env = TurtleBot3Env()
    agent = PPOAgent(env)
    agent.actor_critic.load_state_dict(torch.load('ppo_model.pth'))

    state = env.reset()
    while True:
        action = agent.select_action(state)
        state, _, done, _ = env.step(action)
        if done:
            break
    env.close()

if __name__ == '__main__':
    main()
