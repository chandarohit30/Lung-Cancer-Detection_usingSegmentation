import numpy as np
import random

# Define the environment
num_floors = 4
max_capacity = 3
arrival_probs = np.array([0.2, 0.1, 0.2, 0.5])
departure_probs = np.array([0.25, 0.25, 0.25, 0.25])
actions = ["UP", "DOWN", "WAIT", "PICKUP/DROPOFF"]
num_actions = len(actions)

# Define the Q-table
q_table = np.zeros((num_floors, num_floors, max_capacity+1, num_actions))

# Define the learning parameters
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.1  # Exploration rate

# Define a function to choose an action based on the epsilon-greedy policy
def choose_action(state, epsilon):
    if random.uniform(0, 1) < epsilon:
        # Choose a random action
        return random.randint(0, num_actions-1)
    else:
        # Choose the action with the highest Q-value
        return np.argmax(q_table[state[0], state[1], state[2], :])

# Define a function to simulate the environment and update the Q-table
def simulate(state, action):
    reward = 0
    done = False
    
    # Take the action and update the state
    if action == 0:  # UP
        if state[0] == num_floors - 1:
            next_state = state
        else:
            next_state = (state[0]+1, state[1], state[2])
    elif action == 1:  # DOWN
        if state[0] == 0:
            next_state = state
        else:
            next_state = (state[0]-1, state[1], state[2])
    elif action == 2:  # WAIT
        next_state = state
    elif action == 3:  # PICKUP/DROPOFF
        num_onboard = state[2]
        num_waiting = np.random.binomial(max_capacity - num_onboard, arrival_probs[state[1]])
        num_departing = np.random.binomial(num_onboard, departure_probs[state[0]])
        next_state = (state[0], state[1], num_onboard + num_waiting - num_departing)
        reward = 10 * num_departing - 1 * (num_onboard + num_waiting)
    
    # Check if the task is done
    if state == next_state:
        done = True
    
    return next_state, reward, done


q_value = q_table[state[0], state[1], state[2], action]
next_q_value = np.max(q_table[next_state[0], next_state[1], next_state[2], :])
td_error = reward + gamma * next_q_value - q_value
q_table[state[0], state[1], state[2], action] += alpha * td_error

state = next_state
episode_reward += reward
total_rewards.append(episode_reward)

import matplotlib.pyplot as plt

plt.plot(total_rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.show()
