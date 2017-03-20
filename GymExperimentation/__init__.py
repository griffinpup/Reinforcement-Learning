import gym
import random
import numpy as np
# import pandas as pd
import sys
import itertools

env = gym.make("Pong-v0")
print(env.action_space)
print(env.observation_space)
actionReward = []


# State action value pair
class ActionState:
    def __init__(self, state, action, value=0, count=1):
        self.state = state
        self.action = action
        self.value = value
        self.count = count


def maxAction(state, actionStateList):
    bestAction = -1
    bestValue = 0
    for i in actionStateList:
        if i.state == state:
            if i.value > bestValue:
                bestAction = i.action
                bestValue = i.value
    if bestAction == -1:
        return random.randrange(0, 2)
    else:
        return bestAction


def chooseAction(state, currentIterations):
    decayRate = .99
    endFade = 300
    if random.randrange(0, endFade) < currentIterations:
        return maxAction(state, actionReward)
    else:
        return (random.randint(0, 1))


def updateRewards(actionStatePairs, newActionStates, reward):
    toBeDeleted = []
    print('count')
    print(len(actionStatePairs))
    print(len(newActionStates))
    for newActionIndex, i in enumerate(newActionStates):
        print('UpperLoop')
        for k in actionStatePairs:
            if k.state== i.state and k.action == i.action:
                print(newActionIndex)
                k.value = (k.value * k.count + reward) / (k.count + 1)
                k.count += 1
                toBeDeleted.append(newActionIndex)
    print('lowerLoop')
    for i in toBeDeleted:
        print(i)
        del newActionStates[i]
    for i in newActionStates:
        actionStatePairs.append(ActionState(i.state, i.action))
        actionStatePairs[len(actionStatePairs) - 1].value = reward


for i_episode in range(2000):
    #print(i_episode)
    observation = env.reset()
    episodeStates = [observation]
    episodeActions = []
    episodeReward = 0
    for t in range(100):
        env.render()
        #print(episodeReward)
        action = chooseAction(observation, i_episode)
        observation, reward, done, info = env.step(action)
        episodeStates.append(observation)
        episodeActions.append(action)
        episodeReward += reward
        if done:
            print("Episode finished after {} timesteps".format(t + 1))
            break
    newActionState = []
    for i in episodeActions:
        newActionState.append(ActionState(episodeStates[i], episodeActions[i]))
    updateRewards(actionReward, newActionState, episodeReward)
