import gym
from gym import wrappers
import numpy as np
import random
import scipy


# Calculates the Standard Deviation
def find_SD(index, list):
    coll = []
    for item in list:
        value = item[1][index]
        coll += [value]
    coll = np.array(coll)
    #if np.std(coll) == 0.0:
    #    return 0.1
    return np.std(coll)


env = gym.make("Acrobot-v1")
#env = wrappers.Monitor(env, '/tmp/mountaincar-cross-entropy-1',force=True)

output_size = env.action_space.n
input_size = env.observation_space.high.shape[0]

print output_size
print input_size


# Mean of the deviation
means = []
for i in range(input_size):
    means += [random.random() - 0.5]
means = np.array(means)


# Standard Deviation
standard_deviation = np.array([1] * input_size)

for batch in range(100):

    samples = []
    # The batch
    for iteration in range(100):

        # Choose the current sample

        current_constants = []
        for index, mean in enumerate(means):
            sample = np.random.normal(mean, standard_deviation[index])
            current_constants += [sample]
        current_constants = np.array(current_constants)
        # resets the state
        observation = env.reset()

        observation = np.array(observation)
        total_reward = 0
        done = False

        # The game
        while done == False:
            # env.render()
            # Choose action
            action = current_constants * observation
            action = action.sum()
            if action < 0:
                action = 0
            #if action > output_size-1:
            #    action = output_size-1
            else:
                action = 1
            observation, reward, done, info = env.step(int(action))

            observation = np.array(observation)
            total_reward += reward
        print total_reward
        #print current_constants
        # saves the most recent sample
        samples += [[total_reward, current_constants]]

    '''iterates through the entire sample list, finds the 10 largest, and saves them.'''
    largest_samples = [samples[0]]
    for item in samples:
        for index, sample in enumerate(largest_samples):
            if item[0] > sample[0]:
                if len(largest_samples) > 10:
                    largest_samples[index] = item
                    item = sample
                else:
                    largest_samples += [item]
                    break

    # Mean of the deviation
    means = [0]*input_size


    # Calculates the new mean and standard deviation
    for index in range(len(means)):
        for sample in largest_samples:
            means[index] += sample[1][index]
    means = np.array(means)
    means = means / 11.0
    # Standard Deviation
    for index in range(len(standard_deviation)):
        standard_deviation[index] = find_SD(index, largest_samples)

env.close()
#gym.upload('/tmp/mountaincar-cross-entropy-1', api_key='sk_4cMtKbiGR8SWoX3rEQnVRw')