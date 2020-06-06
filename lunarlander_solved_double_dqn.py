import torch
import torch.nn as nn
import torch.optim as optim
import gym
import random
import numpy as np
import collections
import time
import os
from torch.utils.tensorboard import SummaryWriter

STATE_SIZE = 8
NUM_ACTIONS = 4
MAX_EPISODES = 20000
MAX_STEPS = 3000
LEARNING_RATE = 0.001
DISCOUNT = 0.99
EPSILON = 0.4
TAU = 1.0
SOLVE_THRESHOLD = 200
REPLAY_MEMORY_SIZE = 200000
BATCH_SIZE = 64
PATH_SAVED_MODEL = "models/" + os.path.basename(__file__).split(".")[0] + ".pt"
LOAD_MODEL = False
DEVICE = "cpu"
LEARN_MODE = True
DISPLAY_MODE = True
EXPERIMENT_NAME = "runs/" + "DDQN_" + "batch" + str(BATCH_SIZE) + "_lr_" + str(LEARNING_RATE)

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
    choice = np.random.choice(NUM_ACTIONS, p=p)
    return choice

def policy_egreedy(qa, eps):
    if random.random() < eps:
        return np.random.choice(NUM_ACTIONS)
    else:
        return torch.argmax(qa).item()

def main():
    global TAU, EPSILON, LEARN_MODE

    if not LOAD_MODEL:
        LEARN_MODE = True

    random.seed(10)
    torch.manual_seed(10)
    torch.cuda.manual_seed(10)
    torch.backends.cudnn.benchmark=True

    writer = SummaryWriter(EXPERIMENT_NAME)

    env_name = 'LunarLander-v2'
    env = gym.make(env_name)

    memory = ReplayMemory(REPLAY_MEMORY_SIZE)

    policy_network = nn.Sequential(nn.Linear(STATE_SIZE, 150), nn.ReLU(), nn.Linear(150, 120), nn.ReLU(), nn.Linear(120,NUM_ACTIONS)).to(DEVICE)
    optimizer = optim.Adam (policy_network.parameters(), lr=LEARNING_RATE)
    epoch = 0
    if LOAD_MODEL:
        saved_model = torch.load(PATH_SAVED_MODEL)
        policy_network.load_state_dict(saved_model['model_state_dict'])
        optimizer.load_state_dict(saved_model['optimizer_state_dict'])
        epoch = saved_model['epoch']
        EPSILON = saved_model['epsilon']
        TAU = saved_model['tau']
        policy_network.eval()

    target_network = nn.Sequential(nn.Linear(STATE_SIZE, 150), nn.ReLU(), nn.Linear(150, 120), nn.ReLU(), nn.Linear(120,NUM_ACTIONS)).to(DEVICE)
    target_network.load_state_dict(policy_network.state_dict())
    target_network.eval()

    observation = env.reset()

    reward_last_100 = [] # To calculate the moving average of the last 100
    average_rewards = [] # To store the average in series of 100
    current_average = -9999

    """if DISPLAY_MODE:
        plt.ion()
        plt.xlabel('Episode')
        plt.ylabel('Average reward last 100 episodes')
        plt.title('Lunar Lander')"""

    time_initial = time.time()

    for e in range(epoch, MAX_EPISODES):
        episode_reward = 0
        episode_transitions = []
        episode_loss = 0
        for s in range(MAX_STEPS):
            input = torch.FloatTensor(np.array(observation, copy=False)).to(DEVICE)
            qa = policy_network(input)
            #action = np.random.choice(NUM_ACTIONS, p=qa.detach().numpy())
            #action = policy_softmax(qa, TAU)
            action = policy_egreedy(qa, EPSILON)

            observation, reward, done, _ = env.step(action)

            #print("Step: %s, reward: %s" % (s, reward))
            new_input = torch.FloatTensor(np.array(observation, copy=False)).to(DEVICE)

            if done:
                new_input = None

            episode_transitions.append((input, torch.tensor([action]), new_input, torch.FloatTensor([reward])))


            if e > 10 and len(memory.memory)>=BATCH_SIZE and LEARN_MODE:
                v_states, v_actions, v_newstates, v_rewards = zip(*memory.sample(BATCH_SIZE))

                non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                        v_newstates)), dtype=torch.bool).to(DEVICE)
                non_final_next_states = torch.cat([s for s in v_newstates
                                                   if s is not None]).reshape(-1,8).to(DEVICE)
                v_states = torch.cat(v_states).reshape((BATCH_SIZE,8)).to(DEVICE)
                v_actions = torch.cat(v_actions).reshape(BATCH_SIZE,1).to(DEVICE)
                v_rewards = torch.cat(v_rewards).to(DEVICE)

                qa = policy_network(v_states)
                qa = qa.gather(1, v_actions)
                new_qa = torch.zeros((BATCH_SIZE,1), dtype=torch.float32).to(DEVICE)
                #new_qa[non_final_mask] = target_network(non_final_next_states).max(1)[0]
                #new_qa[non_final_mask] = policy_network(non_final_next_states).max(1)[0] # SAME NETWORK

                # Double DQN
                new_v_actions = torch.zeros((BATCH_SIZE,1), dtype=torch.long).to(DEVICE)
                new_v_actions[non_final_mask] = torch.argmax(policy_network(non_final_next_states),dim=1).reshape((-1,1))
                new_qa[non_final_mask] = target_network(non_final_next_states).gather(1,new_v_actions[non_final_mask])
                new_qa=new_qa.reshape((BATCH_SIZE))

                qa=qa.reshape((BATCH_SIZE))

                target = (v_rewards + DISCOUNT * new_qa).detach()
                loss = nn.MSELoss()(qa, target)
                episode_loss += loss.data
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            episode_reward += reward

            if e% 25 == 0 and DISPLAY_MODE:
                env.render()

            if done:
                break

        episode_loss /= s

        if e%25 == 0 and DISPLAY_MODE:
            print("Episode reward: %s" % episode_reward)

        # Add the episode
        for s in episode_transitions:
            memory.push(*s)

        observation = env.reset()

        # Update parameters
        EPSILON = max(0.01, EPSILON*0.996)
        TAU = max(0.1, TAU*0.995)

        # Load state in target network
        if e>10 and e%1 == 0:
            """for param_target, param_source in zip(target_network.named_parameters(),policy_network.named_parameters()) :
                param_target[1] += 0.95*param_target[1] + 0.05*param_source[1]"""
            target_network.load_state_dict(policy_network.state_dict())
            target_network.eval()
            print(time.time()-time_initial)

        # Calculate averages and display
        reward_last_100.append(episode_reward)
        if len(reward_last_100) > 100:
            reward_last_100 = reward_last_100[-100:]

        current_average = np.mean(reward_last_100)
        average_rewards.append(current_average)
        print("Average reward: %s %s [%s,%s] episode: %s // alpha: . - epsilon: %s - tau: %s" % \
              (current_average, "***SOLVED!***" if current_average > SOLVE_THRESHOLD else "", \
               np.min(reward_last_100), np.max(reward_last_100), e, EPSILON, TAU))

        writer.add_scalar("Avg. reward 100", current_average, e)
        writer.add_scalar("Loss", episode_loss, e)

        # Display the graph
        if e % 10==0 and DISPLAY_MODE:
            #plt.plot(average_rewards, "g")
            #plt.pause(.000001)
            torch.save({
                'epoch': e,
                'model_state_dict': policy_network.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epsilon': EPSILON,
                'tau': TAU
            }, PATH_SAVED_MODEL)

        if current_average > SOLVE_THRESHOLD:
            print("SOLVED IN %s EPISODES WITH AVERAGE REWARD OF %s (LAST 100), MIN=%s, MAX=%s" % (e, current_average, np.min(reward_last_100), np.max(reward_last_100)))
            env.close()
            return

    env.close()

if __name__ == "__main__":
    main()

