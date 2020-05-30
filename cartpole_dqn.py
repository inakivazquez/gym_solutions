import torch
import torch.nn as nn
import torch.optim as optim
import gym
import random
import numpy as np
import matplotlib.pyplot as plt
import collections

class Tools:
    def softmax(x, tau):
        e_x = torch.exp((x - torch.max(x))/tau)
        return e_x / sum(e_x)

Transition = collections.namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

def policy_softmax(qa, tau):
    softmax = Tools.softmax(qa, tau)
    p = softmax.detach().numpy()
    choice = np.random.choice(2, p=p)
    return choice

def policy_egreedy(qa, eps):
    if random.random() < eps:
        return np.random.choice(2)
    else:
        return torch.argmax(qa).item()

def main():
    MAX_EPISODES = 20000
    MAX_STEPS = 200
    DISCOUNT = 1
    EPSILON = 0.8
    TAU = 1
    SOLVE_THRESHOLD = 195
    BATCH_SIZE = 60

    random.seed(10)
    torch.manual_seed(10)

    env_name = 'LunarLander-v2'
    env_name = 'CartPole-v0'
    env = gym.make(env_name)

    memory = ReplayMemory(2000)

    policy_network = nn.Sequential(nn.Linear(4, 120), nn.ReLU(), nn.Linear(120,2))
    target_network = nn.Sequential(nn.Linear(4, 120), nn.ReLU(), nn.Linear(120,2))
    target_network.load_state_dict(policy_network.state_dict())
    optimizer = optim.Adam (policy_network.parameters(), lr=0.0005)

    observation = env.reset()

    reward_last_100 = [] # To calculate the moving average of the last 100
    average_rewards = [] # To store the average in series of 100

    plt.ion()
    plt.xlabel('Episode')
    plt.ylabel('Average reward last 100 episodes')
    plt.title('Cartpole')

    for e in range(MAX_EPISODES):
        episode_reward = 0
        for s in range(MAX_STEPS):
            input = torch.FloatTensor(np.array(observation, copy=False))
            #action = np.random.choice(4, p=qa.detach().numpy())
            qa = policy_network(input)
            #action = policy_softmax(qa, TAU)
            action = policy_egreedy(qa, EPSILON)

            observation, reward, done, _ = env.step(action)

            #print("Step: %s, reward: %s" % (s, reward))
            new_input = torch.FloatTensor(np.array(observation, copy=False))

            if done:
                new_input = None

            memory.push(input, torch.tensor([action]), new_input, torch.tensor([reward]))


            if e > 100:
                v_states, v_actions, v_newstates, v_rewards = zip(*memory.sample(BATCH_SIZE))

                non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                        v_newstates)), dtype=torch.bool)
                non_final_next_states = torch.cat([s for s in v_newstates
                                                   if s is not None]).reshape(-1,4)
                v_states = torch.cat(v_states).reshape((BATCH_SIZE,4))
                v_actions = torch.cat(v_actions).reshape(BATCH_SIZE,1)
                v_rewards = torch.cat(v_rewards)

                qa = policy_network(v_states)
                qa = qa.gather(1, v_actions)
                new_qa = torch.zeros(BATCH_SIZE)
                new_qa[non_final_mask] = target_network(non_final_next_states).max(1)[0]

                qa=qa.reshape((BATCH_SIZE))

                target = (v_rewards + DISCOUNT * new_qa).detach()
                loss = nn.MSELoss()(qa, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            episode_reward += reward

            if e% 100 == 0:
                env.render()

            if done:
                break

        if e== 100:
                EPSILON /= 4
                TAU /= 1
        if e>10 and e%10 == 0:
            target_network.load_state_dict(policy_network.state_dict())
            EPSILON *= 0.99
            TAU *= 0.99

        observation = env.reset()

        reward_last_100.append(episode_reward)
        if len(reward_last_100) > 100:
            reward_last_100 = reward_last_100[1:]

        current_average = np.mean(reward_last_100)
        average_rewards.append(current_average)
        print("Average reward: %s %s [%s,%s] episode: %s // alpha: . - epsilon: %s - tau: %s" % \
              (current_average, "***SOLVED!***" if current_average > SOLVE_THRESHOLD else "", \
               np.min(reward_last_100), np.max(reward_last_100), e, EPSILON, TAU))

        # Display the graph
        if e % 100==0:
            plt.plot(average_rewards, "g")
            plt.pause(.000001)

        if current_average > SOLVE_THRESHOLD:
            print("SOLVED IN %s EPISODES WITH AVERAGE REWARD OF %s (LAST 100), MIN=%s, MAX=%s" % (e, current_average, np.min(reward_last_100), np.max(reward_last_100)))
            env.close()
            return

    env.close()

if __name__ == "__main__":
    main()

