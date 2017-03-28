""" Trains an agent with (stochastic) Policy Gradients on Pong. Uses OpenAI Gym. """
import numpy as np
import cPickle as pickle
import gym

# hyperparameters
neurons = 10  # number of hidden layer neurons
batch_size = 30  # every how many episodes to do a param update?
learning_rate = .6
gamma = 0.99  # discount factor for reward
decay_rate = 0.99  # decay factor for RMSProp leaky sum of grad^2
resume = False  # resume from previous checkpoint?
render = False

# model initialization
env = gym.make("CartPole-v0")
input_size = env.observation_space.high.shape[0]
if resume:
    print("entered if")
    model = pickle.load(open('PongSave.p', 'rb'))
else:
    model = {}
    model['W1'] = np.random.randn(neurons, input_size)  # "Xavier" initialization
    model['W2'] = np.random.randn(neurons)

grad_buffer = {k: np.zeros_like(v) for k, v in model.iteritems()}  # update buffers that add up gradients over a batch
rmsprop_cache = {k: np.zeros_like(v) for k, v in model.iteritems()}  # rmsprop memory

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))  # sigmoid "squashing" function to interval [0,1]

def prepro(I):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    I = I[35:195]  # crop
    I = I[::2, ::2, 0]  # downsample by factor of 2
    I[I == 144] = 0  # erase background (background type 1)
    I[I == 109] = 0  # erase background (background type 2)
    I[I != 0] = 1  # everything else (paddles, ball) just set to 1
    return I.astype(np.float).ravel()

def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(xrange(0, r.size)):
        if r[t] == 0:
            running_add = 0  # reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

observation = env.reset()
prev_x = None  # used in computing the difference frame
processed_observation, hidden_states, gradient, rewards_in_episode = [], [], [], []
running_reward = None
reward_sum = 0
episode_number = 0
while True:

    if episode_number % 1000 == 0:
        env.render()

    # preprocess the observation, set input to network to be difference image
    input = observation

    # forward the policy network and sample an action from the returned probability
    hidden_state = np.dot(model['W1'], input)
    hidden_state[hidden_state < 0] = 0  # ReLU nonlinearity
    logp = np.dot(model['W2'], hidden_state)
    aprob = sigmoid(logp)
    action = 0 if np.random.uniform() < aprob else 1  # roll the dice!

    # record various intermediates (needed later for backprop)
    processed_observation.append(input)  # observation
    hidden_states.append(hidden_state)  # hidden state
    fake_label = 1 if action == 1 else 0  # a "fake label"
    gradient.append(
        fake_label - aprob)  # grad that encourages the action that was taken to be taken (see http://cs231n.github.io/neural-networks-2/#losses if confused)

    # step the environment and get new measurements
    observation, reward, done, info = env.step(action)
    reward_sum += reward

    rewards_in_episode.append(reward)  # record reward (has to be done after we call step() to get reward for previous action)

    if done:  # an episode finished
        episode_number += 1

        # stack together all inputs, hidden states, action gradients, and rewards for this episode
        stored_inputs = np.vstack(processed_observation)
        stored_hidden_states = np.vstack(hidden_states)
        stored_gradients = np.vstack(gradient)
        stored_rewards = np.vstack(rewards_in_episode)
        processed_observation, hidden_states, gradient, rewards_in_episode = [], [], [], []  # reset array memory

        # compute the discounted reward backwards through time
        discounted_rewards = discount_rewards(stored_rewards)
        # standardize the rewards to be unit normal (helps control the gradient estimator variance)
        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= np.std(discounted_rewards)

        stored_gradients *= discounted_rewards  # modulate the gradient with advantage (PG magic happens right here.)

        dW2 = np.dot(stored_hidden_states.T, stored_gradients).ravel()
        dh = np.outer(stored_gradients, model['W2'])
        dh[stored_hidden_states <= 0] = 0  # backpro prelu
        dW1 = np.dot(dh.T, stored_inputs)
        grad = {'W1': dW1, 'W2': dW2}
        for k in model: grad_buffer[k] += grad[k]  # accumulate grad over batch

        # perform rmsprop parameter update every batch_size episodes
        if episode_number % batch_size == 0:
            for k, v in model.iteritems():
                g = grad_buffer[k]  # gradient
                rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g ** 2
                model[k] -= learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
                grad_buffer[k] = np.zeros_like(v)  # reset batch gradient buffer

        # boring book-keeping
        running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
        if episode_number % 100 == 0:
            print 'resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward)
            pickle.dump(model, open('PongSave.p', 'wb'))
            print('saving')
        reward_sum = 0
        observation = env.reset()  # reset env
        prev_x = None

    #if reward != 0:  # Pong has either +1 or -1 reward exactly when game ends.
    #    print ('ep %d: game finished, reward: %f' % (episode_number, reward)) + ('' if reward == -1 else ' !!!!!!!!')