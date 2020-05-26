# -*- coding: utf-8 -*-
"""
Created on Sat May 16 18:22:00 2020

@author: Juan-Ignacio (Inaki) Vazquez (ivazquez@deusto.es)
"""
import numpy as np
import gym
import matplotlib.pyplot as plt
import random

# 

class Tools:
    """ General purpose mathematical tools """
    def softmax(x, tau):
        """ Calculates the softmax of x based on parameter tau """
        e_x = np.exp((x - np.max(x))/tau)
        return e_x / sum(e_x)
    
    def scaler(value, orig_min, orig_max, dest_min, dest_max):
        """ Scales value originally between [orig_min, orig_max] to [dest_min, dest_max] """
        scaled_value = dest_min + (value - orig_min) * (dest_max - dest_min) / (orig_max - orig_min)
        scaled_value = min(max(scaled_value, dest_min), dest_max)
        return scaled_value

class Agent:
    """ An agent that performs action in the environment"""
    def __init__(self, num_inputs, num_actions, num_tilings):
        """
        The init function for the agent

        Args:
            num_inputs: the observation space size
            num_actions: the action space size (discrete)
            tilingset: a tiling set to use
        """
        self.num_inputs = num_inputs
        self.num_actions = num_actions

        self.weights = np.random.random((num_actions, num_inputs))
        self.weights = np.ones((num_actions, num_inputs)) # Can be alternatively initialized with 1's

        self.alpha = 0.4 / num_tilings # According to the number of tilings
        self.discount = 1
        self.epsilon = 0.5 # In the case e-greedy policy is used
        self.tau = 1 # In the case softmax policy is used
        self.lambda_ratio = 0.6
        self.lambda_zeta = 0

    def predict_action_values(self, state, reference_weights=None):
        """
        Calculates the action values for the state by applying the linear function based on the weights
        Args:
            state: the state
            reference_weights: if None the agent weights will be used, but alternative weights can be provided
        """
        if reference_weights is None:
            reference_weights = self.weights
        action_values = np.dot(reference_weights, state.T)
        return action_values


    def policy_e_greedy(self, state):
        """ Implementation of the e-greedy policy based on epsilon """
        if np.random.random() > self.epsilon:
            return np.argmax(self.predict_action_values(state))
        else:
            return np.random.randint(0,self.num_actions)

    def policy_softmax(self, state):
        """ Implementation of the softmax policy based on tau """
        softmax = Tools.softmax(self.predict_action_values(state), self.tau)
        choice = np.random.choice(self.num_actions, p=softmax)
        return choice

    def choose_action(self, state, policy="policy_softmax"):
        """
        Applies the specified policy to obtain the action

        Args:
            state: the state
            policy: the policy to use

        returns:
            the action selected by the policy
        """

        return getattr(self,policy)(state)
    
    def update_weights(self, prev_state, new_state, reward, prev_action, terminal, reference_weights=None):
        """
        Performs the core update process for the linear function weights applying SARSA (lambda)

        Args:
            prev_state: the previous state
            new_state: the new state
            reward: the reward from the transition to new_state
            prev_action: the action that transitioned from prev_prev_state to new_state
            terminal: whether new_state is terminal or not
            reference_weights: optional weights, otherwise the agent's weights will be used
        """
        predicted_action_values = self.predict_action_values(prev_state)
        if not terminal:
            # Apply expected SARSA, to calculate the average value of q for new_state, using softmax
            next_q = np.sum(np.multiply(self.predict_action_values(new_state, reference_weights), \
                                     Tools.softmax(self.predict_action_values(new_state, reference_weights), self.tau)))
            td_error = (reward + self.discount * next_q - predicted_action_values[prev_action])
        else:
            td_error = (reward - predicted_action_values[prev_action])
        self.weights[prev_action] += self.alpha * td_error * self.lambda_zeta
        self.lambda_zeta *= self.discount * self.lambda_ratio


class Tiling:
    """ Represents a uniform tiling """
    def __init__(self, num_dimensions, tiles_per_dimension, thresholds):
        """ Initializes the tiling
        Args:
            num_dimensions: number of dimensions
            tiles per dimension: number of tiles per each dimension (uniform for all the dimensions)
            thresholds: list of threshold values (a tuple of min, max)  for each dimension
            offsets: list of  = array of float, one per dimension

        """
        self.tiles = np.zeros(([tiles_per_dimension] * num_dimensions))
        self.thresholds = thresholds

    def clean(self):
        """ Removes all the data from the tiles """
        self.tiles = np.zeros(self.tiles.shape)

    def set_value(self, multi_value):
        """
        Adds a multidimensional value to the tiling, setting 1 on the tiles where the value is located

        Args:
            multi_value: a list of values, representing a multidimensional vector
        """
        coordinates = []
        for i, v in enumerate(multi_value):
            # Check that the value is between the thresholds for that dimension
            v = max( min( v, self.thresholds[i][1]), self.thresholds[i][0])
            # Calculate the tile position for the value in that dimension
            coor = (v - self.thresholds[i][0]) // ((self.thresholds[i][1] - self.thresholds[i][0]) / self.tiles.shape[i])
            if coor >= self.tiles.shape[i]:
                coor = self.tiles.shape[i]-1
            coordinates.append(int(coor))
        # Finally, set the tile to 1 (feature)
        self.tiles[tuple(coordinates)] = 1

class UniformTilingSet:
    """ A sequence of tilings with offsets """
    def __init__(self, num_tilings, num_dimensions, num_tiles, min, max):
        """
        Initializes the whole structure
        Args:
            num_tilings: number of tilings
            num_dimensions: uniform number of dimensions per tiling
            num_tiles: uniform number of tiles per dimension
            min and max: uniform min and max values for the dimensions
        """
        self.tilings = []
        range_values = max-min
        margin = range_values*0.1
        step = margin / (num_tilings-1)

        # Create the tilings with the appropriate offset
        for i in range(num_tilings):
            t = Tiling(num_dimensions, num_tiles, [[(min-margin+step*i, max+step*i)]*num_dimensions][0])
            self.tilings.append(t)

    def set_value(self, multi_value):
        """ Activates the tile in the proper location according to multi_value in all the tilings """
        for t in self.tilings:
            t.set_value(multi_value)

    def get_features(self):
        """ Returns the unrolled tiles representing the features """
        return np.concatenate(tuple(t.tiles for t in self.tilings), axis=None)

    def clean(self):
        """ Cleans all the tiles in the set"""
        for t in self.tilings:
            t.clean()

def get_features(obs, tiling_set):
    """ Returns the features of an observation of the environment based on the tiling_set"""
    norm_obs = []
    norm_obs.append(Tools.scaler(obs[0], -1.2, 0.6, -1, 1))  # Normalization
    norm_obs.append(Tools.scaler(obs[1], -0.07, 0.07, -1, 1))  # Normalization

    tiling_set.clean()
    tiling_set.set_value(norm_obs)
    return tiling_set.get_features()


def main():
    NUM_TILINGS = 8
    MAX_EPISODES = 2000
    MAX_STEPS = 200
    UPDATE_PARAMETERS_STEPS = 20
    RENDER_EPISODE_EVERY = 50
    SOLVE_THRESHOLD = -110

    random.seed()

    env_name = 'MountainCar-v0'
    env = gym.make(env_name)

    tiling_set = UniformTilingSet(NUM_TILINGS, 2, 10, -1, 1)

    a = Agent(len(get_features(env.observation_space.sample(), tiling_set)),3, NUM_TILINGS)

    reward_last_100 = [] # To calculate the moving average of the last 100
    average_rewards = [] # To store the average in series of 100

    plt.ion()
    plt.xlabel('Episode')
    plt.ylabel('Average reward last 100 episodes')
    plt.title('Mountain car')

    for e in range(MAX_EPISODES):
        episode_reward = 0
        observation = get_features(env.reset(), tiling_set)

        # Initialize the agent episode
        a.lambda_zeta = np.zeros(observation.shape)

        weights = None

        # Change the reference weights every 100 episodes
        if e % UPDATE_PARAMETERS_STEPS == 0:
            weights = np.copy(a.weights)
            if a.alpha > 0.0001: a.alpha *= 0.999 # Decrease alpha, but keep a minimum
            if a.tau > 0.5: a.tau *= 0.999 # Decrease tau, but keep a minimum
            if a.epsilon > 0.001: a.epsilon *= 0.999 # Decrease epsilon, but keep a minimum

        for i in range(MAX_STEPS):
            action = a.choose_action(observation)
            prev_observation = np.copy(observation)
            raw_obs, reward, done, info = env.step(action)
            observation = get_features(raw_obs, tiling_set)

            a.lambda_zeta += prev_observation

            a.update_weights(prev_observation, observation, reward, action, done, weights)

            episode_reward += reward

            # Render one episode every RENDER_STEPS
            if e > 1 and e % RENDER_EPISODE_EVERY == 0:
                env.render()
                if done:
                    print("Render episode reward: %s" % episode_reward)

            if done:
                break;

        reward_last_100.append(episode_reward)
        if len(reward_last_100) > 100:
            reward_last_100 = reward_last_100[1:]

        current_average = np.mean(reward_last_100)
        average_rewards.append(current_average)
        print("Average reward: %s %s [%s,%s] episode: %s // alpha: %s - epsilon: %s - tau: %s" % \
              (current_average, "***SOLVED!***" if current_average > SOLVE_THRESHOLD else "", \
               np.min(reward_last_100), np.max(reward_last_100), e, a.alpha, a.epsilon, a.tau))

        # Display the graph after 100 episodes
        if e > 1:
            plt.plot(average_rewards, "g")
            plt.pause(.000001)

        if current_average > SOLVE_THRESHOLD:
            print("SOLVED IN %s EPISODES WITH AVERAGE REWARD OF %s (LAST 100), MIN=%s, MAX=%s" % (e, current_average, np.min(reward_last_100), np.max(reward_last_100)))
            env.close()
            return

    env.close()
    print("NOT SOLVED IN %s EPISODES" % MAX_EPISODES)

if __name__ == "__main__":
    main()


        
        
        
