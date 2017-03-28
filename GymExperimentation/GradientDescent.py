import gym
from gym import wrappers
import numpy as np
import random
import scipy

def sigmoid(x, deriv = False):
    if (deriv == True):
        return x*(1-x)
    return 1/(1+np.exp(-x))

env = gym.make("CartPole-v0")
#env = wrappers.Monitor(env, '/tmp/mountaincar-cross-entropy-1',force=True)

#print env.action_space
#print env.observation_space

output_size = env.action_space.n
input_size = env.observation_space.high.shape[0]

print output_size
print input_size

#Parameters
node_count = 30
batch_size = 100
number_of_batches = 1000

#2 layers of the neural network
layer1 = 2 * np.random.random((input_size, node_count)) - 1
layer2 = 2 * np.random.random((node_count, output_size)) - 1




for batch in range(number_of_batches):


    for iteration in range(batch_size):

        # reset game level statistics
        observation = env.reset()
        observation = np.array(observation)
        episode_gradient_log = []
        hidden_states = []
        stored_inputs = []
        stored_rewards = []
        total_reward = 0
        done = False

        # The game
        while done == False:
            #env.render()

            #The forward pass of the neural network
            l1 = sigmoid(np.dot(observation, layer1))
            l2 = sigmoid(np.dot(l1, layer2))

            #Selecting one of the actions, based on relative probability
            denominator = l2.sum()
            if random.random() < l2[0] / denominator:
                action = 0
            else:
                action = 1

            #Update
            observation, reward, done, info = env.step(action)
            total_reward += reward

            #Calculates the error: Initializes an array, sets the current action to 1, and all others to 0, then
            #subtracts by the output values.
            loss_function_gradient = np.zeros(l2.shape)
            loss_function_gradient[action] = 1
            loss_function_gradient = loss_function_gradient - l2
            loss_function_gradient = loss_function_gradient * sigmoid(l2, deriv=True)
            print sigmoid(l2, deriv=True)
            #Saves this timestep's loss function
            episode_gradient_log.append(loss_function_gradient)
            #Saves this timestep's observed state
            stored_inputs.append(observation)
            #saves this timestep's hidden state
            hidden_states.append(l1)
            #saves this timestep's reward
            stored_rewards.append(reward)

        print total_reward
        #Unstack all of the data from the most recent game
        gradient_log = np.vstack(episode_gradient_log)
        input_log = np.vstack(stored_inputs)
        states_log = np.vstack(hidden_states)
        reward_log = np.zeros(len(stored_rewards))

        #Reset the values of all the game-level holders
        episode_gradient_log, stored_inputs, hidden_states, stored_rewards= [], [], [], []

        #Scales the gradient from the most recent game by the discounted reward
        for index in reversed(range(0, len(reward_log)-1)):
            total_reward = total_reward * .99
            reward_log[index] = total_reward

        #Averages the reward
        #reward_log -= np.mean(reward_log)
        #reward_log /= np.std(reward_log)

        #multiplies the reward by the gradients
        for index, reward in enumerate(reward_log):
            for gradient_index in range(len(gradient_log[index])):
                gradient_log[index][gradient_index] *= reward

        #layer2_delta = np.dot(states_log.T, gradient_log)

        layer1_error = gradient_log.dot(layer1.T)
        layer1_delta = layer1_error * sigmoid()
        #height_delta = np.dot(gradient_log, layer2.T)
        #layer1_delta = np.dot(input_log.T, height_delta)

        layer2 += gradient_log
        layer1 += layer1_delta

#Discounts the rewards
discounted_rewards = np.zeros(reward_log)
running_reward = 0
for i in reversed(range(0, discounted_rewards.size)):
    running_reward = running_reward * .99 + discounted_rewards[i]
    discounted_rewards[i] = running_reward
discounted_rewards -= np.mean(discounted_rewards)



env.close()
#gym.upload('/tmp/mountaincar-cross-entropy-1', api_key='sk_4cMtKbiGR8SWoX3rEQnVRw')