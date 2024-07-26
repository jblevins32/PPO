import matplotlib.pyplot as plt
import numpy as np

# Define the functions
def goal_reward_function(x):
    return 50 / np.exp(np.abs(x))

def collision_punishment_function(x):
    return -10 / np.exp(np.abs(x))

# Generate x values
x = np.linspace(-10, 10, 400)

# Generate y values
y_goal_reward = goal_reward_function(x)
y_collision_punishment = collision_punishment_function(x)

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(x, y_goal_reward, label='Goal Reward Function', color='blue')
plt.plot(x, y_collision_punishment, label='Collision Punishment Function', color='red')
plt.xlabel('Distance')
plt.ylabel('Reward')
plt.legend()
plt.grid(True)
plt.show()

hold = 0
