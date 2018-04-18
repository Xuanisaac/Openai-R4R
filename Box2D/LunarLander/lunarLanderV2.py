import gym
import numpy as np
from keras.optimizers import Adam
from keras import backend as K

import random

import os
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *

seed = 16
env = gym.make('LunarLander-v2')
np.random.seed(seed)
random.seed(seed)
env.seed(seed)
env = gym.wrappers.Monitor(env, './monitor/', force=True)

def huber_loss(y_true, y_pred):
    err = y_pred - y_true
    return K.mean(K.sqrt(1+K.square(err))-1, axis=-1 )

class Experience(object):
    def __init__(self, maxsize=500000):
        self.maxsize = maxsize
        self.state  = []
        self.action  = []
        self.reward  = []
        self.state1 = []
        self.done = []
        self.curr_size = 0

    def save(self, s, a, r, s1, _done):
        self.state.append(s)
        self.action.append(a)
        self.reward.append(r)
        self.state1.append(s1)
        self.done.append(_done)
        self.curr_size += 1

        if self.curr_size > self.maxsize:
            self.state.pop(0)
            self.action.pop(0)
            self.reward.pop(0)
            self.state1.pop(0)
            self.done.pop(0)
            self.curr_size -= 1

    def sample(self, n):
        N = self.curr_size
        n = min(n, N)
        idx = np.random.choice(N, n, replace=False)
        s    = [self.state[i]  for i in idx]
        a    = [self.action[i] for i in idx]
        r    = [self.reward[i] for i in idx]
        s1   = [self.state1[i] for i in idx]
        done = [self.done[i]   for i in idx]
        return (s, a, r, s1, done)


class SolutionAgent(object):

    def __init__(self, state_dim, action_dim, decay=0.985,
                buffersize=500000, samplesize=64, minsamples=10000,
                gamma=0.99,updateTargetFreq=600):

        self.state_dim  = state_dim
        self.action_dim = action_dim
        self.exp        = Experience(maxsize=buffersize)
        self.epsilon    = 1.0
        self.minsamples = minsamples
        self.samplesize = samplesize
        self.decay      = decay
        self.gamma      = gamma
        self.updateTargetFreq = updateTargetFreq
        self.steps = 0
        self.model        = self.createModel()
        self.target_model = self.createModel()

    def createModel(self):
        model = Sequential()
        model.add(Dense(50, input_dim=self.state_dim))
        model.add(Activation("relu"))
        model.add(Dense(40))
        model.add(Activation("relu"))
        model.add(Dense(self.action_dim))
        model.add(Activation('linear'))
        model.compile(loss=huber_loss, optimizer=Adam(lr=0.0005))
        return model

    def actionSelection(self, curState, testmode=False):
        if not testmode:
            if (np.random.random() <= self.epsilon):
                return random.randint(0, self.action_dim-1)
        s = np.array([curState])
        return np.argmax(self.model.predict(s)[0])

    def evaluate(self, prevState, action, reward, curState, done):
        self.exp.save(prevState, action, reward, curState, done)
        if done:
            self.epsilon *= self.decay
        if self.steps % self.updateTargetFreq == 0:
            self.target_model.set_weights(self.model.get_weights())
        self.steps += 1

    def learn(self):
        if self.exp.curr_size <= self.minsamples:
            return 0.0
        X, y = self.get_training_data()
        loss = self.model.train_on_batch(X,y)
        return loss

    def get_training_data(self):
        s, a, r, s1, done = self.exp.sample(self.samplesize)
        s  = np.array(s)
        s1 = np.array(s1)
        q  = self.model.predict(s)
        q1 = self.target_model.predict(s1)
        X = s
        y = np.zeros((self.samplesize, self.action_dim))
        for idx in xrange(self.samplesize):
            reward = r[idx]
            action = a[idx]
            target = q[idx]
            target_for_action = reward
            if not done[idx]:
                target_for_action += ( self.gamma*max(q1[idx]) )
            target[action] = target_for_action
            y[idx, :] = target
        return X, y



agent = SolutionAgent(state_dim=env.observation_space.shape[0], action_dim=env.action_space.n,
                    decay=0.985, buffersize=500000, samplesize=32, minsamples=1000,
                    gamma=0.99, updateTargetFreq=600)


print( "-------------------------Training Mode-----------------------")
numOfEpisodes = 1000
testmode = False
window_avg = [0] * 100
max_mean = 0.0
train_file = open("train.csv", "wb")
for idx in xrange(numOfEpisodes):
    currSatet = env.reset()
    done = False
    loss = 0.0
    numSteps = 0
    total_reward = 0.0

    while True:
        numSteps += 1
        action = agent.actionSelection(currSatet, testmode=False)
        prevState = currSatet
        currSatet, reward, done, info = env.step(action)

        total_reward += reward

        if not testmode:
            agent.evaluate(prevState, action, reward, currSatet, done)
            loss += agent.learn()
        if done: break

    window_avg.pop(0)
    window_avg.append(total_reward)
    curmean = np.mean(window_avg)
    if curmean > max_mean:
        max_mean = curmean
    # if max_mean > 200: break
    print ('Episode NO.{%d}:  Reward = {%.2f}, Mean: {%.2f} Loss = {%f}, Steps = {%d}'.format(idx+1,total_reward,curmean,loss,numSteps))
    train_file.write(total_reward)
    train_file.write('\n')

train_file.close()
print ("------------End of Training:   max mean {%.2f}-----------------".format(max_mean))
print ("-------------------------Test Mode-----------------------")

numOfEpisodes = 1000
testmode = True
test_file = open("test.csv", "wb")

for idx in xrange(numOfEpisodes):
    currSatet = env.reset()
    done = False
    loss = 0.0
    numSteps = 0
    total_reward = 0.0

    while True:
        numSteps += 1
        action = agent.actionSelection(currSatet, testmode)
        prevState = currSatet
        currSatet, reward, done, info = env.step(action)
        total_reward += reward
        if done: break
    test_file.write(str(total_reward))
    test_file.write('\n')
    print ("Reward: {%.2f}".format(total_reward))

env.close()

# gym.upload('./monitor', api_key="")


