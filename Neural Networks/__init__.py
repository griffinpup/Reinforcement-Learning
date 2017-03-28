import numpy as np
import gym

import random

#Makes the number 'squished' between 0 and 1, or finds the derivative of that number
def sigmoid(array, deriv=False):
    if (deriv == True):
        return array * (1 - array)

    return 1 / (1 + np.exp(-array))

def rectifier(array, deriv=False):

    if (deriv == True):
        return 1

    entered = False
    for index in range(len(array[0])):
        entered = True
        if array[0][index] < 0:
            array[0][index] = 0

    return array

env = gym.make("CartPole-v0")
print(env.action_space)
print(env.observation_space)

#Parameters
batch_size = 3
node_count = 20
decay_rate = .99
learning_rate = .001
#output_size = env.action_space.n
#input_size = env.observation_space.high.shape[0]
output_size = 1
input_size = 1
batch_count = 0

#The two neuron layers,
# INPUTxNODE_COUNT
layer1 = 2 * np.random.random((input_size, node_count)) - 1
# NODE_COUNTxOUTPUT
layer2 = 2 * np.random.random((node_count, output_size)) - 1

obs = env.reset()

batch_errors = []
batch_l1 = []
batch_l2 = []

while True:
    batch_count+= 1
    obs = random.random()
    l1 = sigmoid(np.dot(obs, layer1))
    l2 = sigmoid(np.dot(l1, layer2))



    # how much did we miss the target value?
    l2_error = obs - l2

    batch_errors.append(l2_error)
    batch_l1.append(l1)
    batch_l2.append(l2)

    if batch_count % batch_size == 0:
        batch_errors = np.vstack(batch_errors)
        batch_l1 = np.vstack(batch_l1)
        batch_l2 = np.vstack(batch_l2)
        for index, error in enumerate(batch_errors):
            l1_state = np.vstack(batch_l1[index])
            l2_state = np.vstack(batch_l2[index])
            # in what direction is the target value?
            # were we really sure? if so, don't change too much.
            l2_delta = error * sigmoid(l2_state, deriv=True)
            # how much did each l1 value contribute to the l2 error (according to the weights)?
            l1_error = l2_delta.dot(layer1)

            # in what direction is the target l1?
            # were we really sure? if so, don't change too much.
            l1_delta = l1_error * sigmoid(l1_state, deriv=True)
            temp_var = l1_state.dot(l2_delta)

            layer2 += l1_state.dot(l2_delta)
            layer1 += l2_state.dot(l1_delta)
        batch_errors = []
        batch_l1 = []
        batch_l2 = []

