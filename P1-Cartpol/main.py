import gym
import numpy as np
env = gym.make('CartPole-v0')
env.reset()

class Solution():
    def __init__(self, env):
        self.env = env
    
    def run_episode(self):
        observation = env.reset()
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


sol = SolRandom(env)
print(sol.run())


sol = SolHillClimb(env)
print(sol.run())
# for _ in range(1000):
#     env.render()
#     env.step(env.action_space.sample()) # take a random action
