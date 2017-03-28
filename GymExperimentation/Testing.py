import gym
import random
import numpy as np
# import pandas as pd
import sys
import itertools

#Makes the number 'squished' between 0 and 1, or finds the derivative of that number
def sigmoid(x, deriv=False):
    if (deriv == True):
        return x * (1 - x)

    return 1 / (1 + np.exp(-x))

#Backpropogates the gradient with an RMS (Root mean Squared) algorithm
def backpropogate(gradient, layer, rms_layer):
    rms_layer = decay_rate * rms_layer + (1 - decay_rate) * gradient ** 2
    new_gradient = learning_rate * gradient / (np.sqrt(rms_layer) + 1e-4)
    layer += new_gradient
    return rms_layer, layer


def get_test_obs(action, given_input):
    obs = random.randint(0,10)
    reward = action-given_input

    return float(obs), reward, True, "garbage"

def test_reset():
    return float(random.randint(0,10))

def test_backpropogate(gradient, layer, rms_layer):
    layer += gradient
    return layer


#initialized the environment and prints pertinent info
env = gym.make("CartPole-v0")
print(env.action_space)
print(env.observation_space)

#Parameters
batch_size = 30
node_count = 2
decay_rate = .99
learning_rate = .001
output_size = env.action_space.n
input_size = env.observation_space.high.shape[0]
output_size = 1
input_size = 1

#The two neuron layers,
# INPUTxNODE_COUNT
layer1 = 2 * np.random.random((input_size, node_count)) - 1
# NODE_COUNTxOUTPUT
layer2 = 2 * np.random.random((node_count, 1)) - 1

#Initializes the lists that contain RMS history
rmsprop_l2 = np.zeros_like(layer2)
rmsprop_l1 = np.zeros_like(layer1)

#
#observation = env.reset()
observation = test_reset()
#observation = np.array(observation)

#Initialize batch level lists
batch_rewards = []
batch_errors = []
batch_l1_output = []
batch_l2_output = []
batch_inputs = []
episode_count = 0

#Initialize game level lists
error = []
l1_output = []
l2_output = []
inputs = []
total_reward = 0


while True:
    #Push forward through the sigmoid
    l1 = sigmoid(np.dot(observation, layer1))
    l2 = sigmoid(np.dot(l1, layer2))

    #choose the action based on the neuron's output.
    action = l2

    print "Observation: " + str(observation)
    print "Action: " + str(l2)

    #Play the game
    #observation, reward, done, info = env.step(action)
    observation, reward, done, info = get_test_obs(action, observation)

    #if episode_count % 100 == 0:
        #env.render()

    #Save all the pertinent data
    #ITERATIONSx1
    error.append(action - l2)
    #ITERATIONSxNODE_COUNT
    l1_output.append(l1)
    #ITERATIONSx1
    l2_output.append(l2)
    #ITERATIONSxINPUT_COUNT
    inputs.append(observation)
    total_reward += reward

    #If the game is ended
    if done:
        #print total_reward
        #Stacks all of the data to make it more manageable
        error = np.vstack(error)
        l1_output = np.vstack(l1_output)
        l2_output = np.vstack(l2_output)
        inputs = np.vstack(inputs)
        print "Error: " + str(error)

        episode_count += 1

        #Add all of the game data to the batch-level data
        batch_rewards.append(total_reward)
        batch_errors.append(error)
        batch_l1_output.append(np.vstack(l1_output))
        batch_l2_output.append(np.vstack(l2_output))
        batch_inputs.append(np.vstack(inputs))

        #Resets all of the game-level information
        env.reset()
        error = []
        l1_output = []
        l2_output = []
        inputs = []
        total_reward = 0

        #If the entire batch is done
        if episode_count % batch_size == 0:
            #Stacks to data to make it easier to work with, and makes everything a numpy array
            batch_rewards = np.vstack(batch_rewards)
            batch_l2_output = np.array(batch_l2_output)
            batch_l1_output = np.array(batch_l1_output)
            batch_inputs = np.array(batch_inputs)
            batch_errors = np.array(batch_errors)
            val = np.mean(batch_rewards)
            batch_rewards = np.ndarray.astype(batch_rewards, float)
            batch_rewards -= np.mean(batch_rewards)

            print "Batch Errors" + str(batch_errors)
            #Iterate through every award
            for index, reward in enumerate(batch_rewards):
                #Multiply a discounted reward through the length of the game
                for frame in reversed(range(0, len(batch_errors[index]))):
                    batch_errors[index][frame] = batch_errors[index][frame] * reward * .99
                    reward = reward * .99

                #calculate the gradient for the second layer
                #ITERATIONSx1:  the error * the slope
                l2_delta = batch_errors[index] * sigmoid(batch_l2_output[index], deriv=True)
                #print "sigmoid(batch_l2_output[index], deriv=True)" + str(sigmoid(batch_l2_output[index], deriv=True))
                #print "batch_errors[index]" + str(batch_errors[index])
                #print "l2_delta" + str(l2_delta)
                #Multiplies the delta by layer 2's input
                #NODE_COUNTxOUTPUT
                final_gradient_l2 = l2_delta.T.dot(batch_l1_output[index])
                final_gradient_l2 = final_gradient_l2.T
                #print "Final Gradient l2" + str(final_gradient_l2)
                #Updates layer2 and RMS memory
                #rmsprop_l2, layer2 = backpropogate(final_gradient_l2, layer2, rmsprop_l2)

                layer2 = test_backpropogate(final_gradient_l2, layer2, 10)

                #rmsprop_l2 = decay_rate * rmsprop_l2 + (1-decay_rate) * final_gradient_l2 ** 2
                #layer2 += learning_rate * final_gradient_l2 / (np.sqrt(rmsprop_l2) + 1e-4)

                #ITERATIONSxNODE_COUNT
                l1_error = l2_delta.dot(layer2.T)
                #calculates the gradient of the first layer
                #IterationsxNODE_COUNT: the error * the slope
                l1_delta = l1_error * sigmoid(batch_l1_output[index], deriv=True)
                # Multiplies the delta by layer 1's input
                # INPUT_COUNTxNODE_COUNT
                final_gradient_l1 = l1_delta.T.dot(batch_inputs[index])
                final_gradient_l1 = final_gradient_l1.T
                #Updates layer1 and RMS memory
                #rmsprop_l1, layer1 = backpropogate(final_gradient_l1, layer1, rmsprop_l1)
                layer1 = test_backpropogate(final_gradient_l1, layer1, 10)
                #layer1 += final_gradient_l1

                print

            #reset containers
            batch_rewards = []
            batch_l1_output = []
            batch_l2_output = []
            batch_rewards = []
            batch_errors = []
            batch_inputs = []