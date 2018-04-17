import gym
import numpy as np
import tensorflow as tf
import random

env = gym.make('CartPole-v0')
env.monitor.start('cartpole/', force=True)
env.reset()

class Solution():
    def __init__(self, env):
        self.env = env
    
    def run_episode(self):
        observation = self.env.reset()
        totalreward = 0
        for _ in range(200):
            action = self.obs_to_action(observation)
            observation, reward, done, info = env.step(action)
            totalreward += reward
            if done:
                break
        return totalreward   

    def obs_to_action(self, observation):
        return env.action_space.sample()

class SolRandom(Solution):
    def __init__(self, env):
        super().__init__(env)
        self.parameters = np.random.rand(4) * 2 -1

    def obs_to_action(self, observation):
        action = 0 if np.matmul(self.parameters, observation) < 0 else 1
        return action

    def run(self):
        bestparams = None
        bestreward = 0
        for laps in range(10000):
            self.parameters = np.random.rand(4) *2 -1
            reward = self.run_episode()
            if reward > bestreward:
                bestparams = self.parameters
                if reward == 200:
                    print("solved")
                    return laps, self.parameters

# TODO : improve hill climbing
class SolHillClimb(Solution):
    def __init__(self, env):
        super().__init__(env)
        self.parameters = np.random.rand(4) * 2 -1
        self.bestparams = self.parameters
        self.alpha = 0.1

    def obs_to_action(self, observation):
        action = 0 if np.matmul(self.parameters, observation) < 0 else 1
        return action

    def run(self):
        bestreward = 0
        for laps in range(10000):
            self.parameters = self.bestparams + (np.random.rand(4) *2 -1) * self.alpha
            reward = self.run_episode()
            if reward > bestreward:
                self.bestparams = self.parameters
                bestreward = reward
                if reward == 200:
                    print("solved")
                    return laps, self.parameters

# Finish Policy Gradient
class SolPolicyGrad(Solution):
    def __init__(self, env):
        super.__init__(env)

    def run_episode(self):
        pl_calculated, pl_state, pl_action,  pl_adv, pl_optimizer = policy_grad
        vl_calculated, vl_state, vl_newvals, vl_optimizer, vl_loss = value_grad
        obs = self.env.reset()
        totalreward = 0
        states = []
        actions = []
        advs = []
        transitions = []
        update_vals = []

        for _ in range(200):
            # calculate policy
            obs_vector = np.expand_dims(obs, axis=0)
            probs = sess.run(pl_calculated, feed_dict={pl_state: obs_vector})
            action = 0 if random.uniform(0, 1) < probs[0][0] else 1
            # record the transition
            states.append(obs)
            actionblank = np.zeros(2)
            actionblank = 1
            actions.append(actionblank)

            # take the action in env
            old_obs = obs
            obs, reward, done, info = env.step(action)
            totalreward += reward

            if done:
                break

        for index, trans in enumerate(transitions):
            obs, action, reward = trans

            future_reward = 0
            future_transitions = len(transitions) - index
            decrease = 1

            for index2 in range(future_transitions):
                future_reward += transitions[(index2) + index][2] * decrease
                decrease = decrease * 0.97

            obs_vector = np.expand_dims(obs, axis=0)
            currentval = sess.run(vl_calculated, feed_dict={vl_state: obs_vector})[0][0]

            advs.append(future_reward - currentval)
            update_vals.append(future_reward)

        update_vals_vector = np.expand_dims(update_vals, axis=1)
        sess.run(vl_optimizerm, feed_dict={vl_state: states, vl_newvals: update_vals_vector})

        advs_vector = np.expand_dims(advs, axis=1)
        sess.run(pl_optimizerm, feed_dict={pl_state: states, pl_adv: advs_vector, pl_action: actions})

        return totalreward


    def run(self):
        policy_grad = policy_gradient()
        value_grad = value_gradient()
        sess = tf.InteractiveSession()
        sess.run(tf.initialize_all_variables())

        for i in range(2000):
            reward = run_episode(env, policy_grad, value_grad, sess)
            if reward > 200:
                print( "reward 200")
                print i 
                return i
        
        t = 0
        for _ in range(1000):
            reward = run_episode(env, policy_grad, value_grad, sess)
            t += reward
        
        return float(t)/1000

def policy_gradient():
    with tf.variable_scope("policy"):
        params = tf.get_variable("policy_parameters", [4,2])
        state = tf.placeholder("float", [None, 4])
        actions = tf.placeholder("float", [None, 2])
        
        adv = tf.placeholder("float", [None, 1])

        linear = tf.matmul(state, params)
        probs = tf.nn.softmax(linear)
        good_probs = tf.reduce_sum(tf.mul(probs, actions), reduction_indices=[1])
        
        log_probs = tf.log(good_probs)
        
        eligibility =  log_probs * adv
        loss = -tf.reduce_sum(eligibility)
        optimizer = tf.train.AdamOptimizer(0.01).minimize(loss)

        return probs, state, action, adv, optimize

def value_gradient():
    with tf.variable_scope("value"):
        state = tf.placeholder("float", [None, 4])
        w1 = tf.get_variable("w1", [4, 10])
        b1 = tf.get_variable("b1", [10])
        h1 = tf.nn.relu(tf.matmul(state, w1) + b1)

        w2 = tf.get_variable("w2", [10, 1])
        b2 = tf.get_variable("b2", [1])
        calculated = tf.matmul(h1, w2) + b2
        newvals = tf.placeholder("float", [None, 1])
        diffs = calculated - newvals
        loss = tf.nn.l2_loss (diffs)

        optimizer = tf.train.AdamOptimizer(0.1).minimize(loss)

        return calculated, state, newvals, optimizer, loss


sol = SolRandom(env)
print(sol.run())


sol = SolHillClimb(env)
print(sol.run())
# for _ in range(1000):
#     env.render()
#     env.step(env.action_space.sample()) # take a random action

env.monitor.close()