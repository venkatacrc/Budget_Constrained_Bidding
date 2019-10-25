import sys,os
sys.path.append(os.getcwd()+'/src/gym-auction_emulator')
import gym, gym_auction_emulator
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

import configparser
from dqn import Agent
from reward_net import RewardNet
import numpy as np


class RlBidAgent():

    def _load_config(self):
        """
        Parse the config.cfg file
        """
        cfg = configparser.ConfigParser(allow_no_value=True)
        env_dir = os.path.dirname(__file__)
        cfg.read(env_dir + '/config.cfg')
        self.budget = int(cfg['agent']['budget'])
        self.target_value = int(cfg['agent']['target_value'])
        self.T = int(cfg['rl_agent']['T']) # T number of timesteps
        self.STATE_SIZE = int(cfg['rl_agent']['STATE_SIZE'])
        self.ACTION_SIZE = int(cfg['rl_agent']['ACTION_SIZE'])
        
    
    def __init__(self):
        self._load_config()
        # Control parameter used to scale bid price
        self.BETA = [-0.08, -0.03, -0.01, 0, 0.01, 0.03, 0.08]
        self.eps_start = 0.95
        self.eps_end = 0.05
        self.anneal = 0.00005
        self._reset_episode()
        # DQN Network to learn Q function
        self.dqn_agent = Agent(state_size = 7, action_size=7, seed =0)
        # Reward Network to reward function
        self.reward_net = RewardNet(state_action_size = 8, reward_size=1, seed =0)
        self.dqn_state = None
        self.dqn_action = 3 # no scaling
        self.dqn_reward = 0
        # Reward-Dictionary
        self.reward_dict = {}
        self.S = []
        self.V = 0
        self.total_wins = 0
        self.total_rewards = 0.0

    def _reset_episode(self):
        """
        Function to reset the state when episode changes
        """
        self.t_step = 0                   # 1. t: the current time step
        self.budget_spend = 0.0
        self.rem_budget = self.budget     # 2. the remaining budget at time-step t
        self.ROL = self.T                 # 3. the number of Lambda regulation opportunities left
        self.prev_budget = self.budget    # Bt-1
        self.BCR = 0                      # 4. Budget consumption rate
                                          #      (self.budget - self.prev_budget) / self.prev_budget
        self.CPM = 0                      # 5. Cost per mille of impressions between t-1 and t
                                          #       (self.prev_budget - self.running_budget) / self.cur_wins
        self.WR = 0                       # 6. wins_e / total_impressions
        self._reset_step()                # 7. Total value of the winning impressions 'click_prob'
        self.cur_day = 1
        self.cur_hour = 0
        self.ctl_lambda = 1.0  # Lambda sequential regulation parameter
        self.wins_e = 0  
        self.eps = self.eps_start
        self.V = 0

    def _update_step(self):
        """
        Function to call to update the state with every bid request
        received for the state modeling
        """
        self.t_step += 1
        self.prev_budget = self.rem_budget
        self.rem_budget -= (self.cost_t / 1e9)
        self.ROL -= 1
        self.BCR = (self.rem_budget - self.prev_budget) / self.prev_budget
        self.CPM = self.cost_t
        self.WR = self.wins_t / self.bids_t

    def _reset_step(self):
        """
        Function to call every time a new time step is entered.
        """
        self.reward_t = 0.
        self.cost_t = 0.
        self.wins_t = 0
        self.bids_t = 0
        self.eps = max(self.eps_start - self.anneal * self.t_step, 0.05)
    
    def _update_reward_cost(self, reward, cost):
        """
        Internal function to update reward and action to compute the cumulative
        reward and cost within the given step.
        """
        self.reward_t += reward
        self.cost_t += cost
        self.bids_t += 1
        self.total_rewards += reward
        
    def _get_state(self):
        """
        Returns the state that will be used for the DQN state.
        """
        return np.asarray([self.t_step,
                self.rem_budget,
                self.ROL,
                self.BCR,
                self.CPM,
                self.WR,
                self.reward_t])

    def act(self, state, reward, cost):
        """
        This function gets called with every bid request.
        By looking at the weekday and hour to progress between the steps and
        episodes during training.
        Returns the bid request cost based on the scaled version of the
        bid price using the DQN agent output.
        """
        episode_done = (state['weekday'] != self.cur_day)
        # within the time step
        if state['hour'] == self.cur_hour and state['weekday'] == self.cur_day:
            self._update_reward_cost(reward, cost)
        # within the episode, changing the time step
        elif state['hour'] != self.cur_hour and state['weekday'] == self.cur_day:
            self._update_step()
            # Sample a mini batch and perform grad-descent step
            self.reward_net.step()
            dqn_next_state = self._get_state()
            a_beta = self.dqn_agent.act(dqn_next_state, eps=self.eps)
            sa = np.append(self.dqn_state, self.dqn_action)
            rnet_r = float(self.reward_net.act(sa))
            # call agent step
            self.dqn_agent.step(self.dqn_state, self.dqn_action, rnet_r, dqn_next_state, episode_done)
            self.dqn_state = dqn_next_state
            self.dqn_action = a_beta
            # print(dqn_next_state, a_beta)
            self.ctl_lambda *= (1 + self.BETA[a_beta])
            self.cur_hour = state['hour']
            self._reset_step()
            self._update_reward_cost(reward, cost)
            self.V += self.reward_t
            self.S.append((self.dqn_state, self.dqn_action))
        # episode changes
        elif state['weekday'] != self.cur_day:
            for (s, a) in self.S:
                sa = tuple(np.append(s, a))
                max_r = max(self.reward_net.get_from_M(sa), self.V)
                self.reward_net.add_to_M(sa, max_r)
                self.reward_net.add(sa, max_r)
            print("Total Impressions won with Budget={} Spend={} wins = {}".format(self.budget, self.budget_spend, self.wins_e))
            self.total_wins += self.wins_e
            self._reset_episode() 
            self.cur_day = state['weekday']
            self.cur_hour = state['hour']
            self._update_reward_cost(reward, cost)

        # action = bid amount
        # send the best estimate of the bid
        self.budget_spend += (cost / 1e9)
        if cost > 0:
            self.wins_t += 1
            self.wins_e += 1
        action = min(self.ctl_lambda * self.target_value * state['click_prob'] * 1e9,
                        (self.budget - self.budget_spend) * 1e9)
        return action

    def done(self):
        return self.budget <= self.budget_spend

def main():
    # Instantiate the Environment and Agent
    env = gym.make('AuctionEmulator-v0')
    env.seed(0)
    agent = RlBidAgent()

    obs, reward, cost, done = env.reset()
    agent.cur_day = obs['weekday']
    agent.cur_hour = obs['hour']
    agent.dqn_state = agent._get_state()

    while not done:
        # action = bid amount
        action = agent.act(obs, reward, cost)
        next_obs, reward, cost, done = env.step(action)
        obs = next_obs # Next state assigned to current state
        # done = agent.done()

    print("Total Impressions won {} value = {}".format(agent.total_wins, agent.total_rewards))
    env.close()

if __name__ == "__main__":
    main()
