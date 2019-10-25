"""
Run this module inside Budget_Constrained_Bidding directory
%python3 src/rtb_agent/constant_bid_agent.py
"""
import sys,os
sys.path.append(os.getcwd()+'/src/gym-auction_emulator')
import gym, gym_auction_emulator

import configparser

"""
Simple toy constant bidding agent that constantly bids $1 until budget runs out
This is an example to show the OpenAI gym interface for the 
    Budget Constrained Bidding problem.
"""

class LinearBidAgent():

    def _load_config(self):
        """
        Parse the config.cfg file
        """
        cfg = configparser.ConfigParser(allow_no_value=True)
        env_dir = os.path.dirname(__file__)
        cfg.read(env_dir + '/config.cfg')
        self.budget = int(cfg['agent']['budget'])
        self.target_value = int(cfg['agent']['target_value'])
 
    def __init__(self):
        self._load_config()
        self.wins_e = 0 # wins in each episode
        self.total_wins = 0 # total wins 
        self.total_rewards = 0.0
        self.budget_spend = 0
        self.cur_day = 1

    def act(self, state, reward, cost):

        # episode done
        if state['weekday'] != self.cur_day:
            print("Total Impressions won with Budget={} Spend={} wins = {}".format(self.budget, self.budget_spend, self.wins_e))
            # reallocate budget
            self.budget_spend = 0
            # reset episode wins
            self.wins_e = 0
            self.cur_day = state['weekday']
            

        # action = bid amount
        # send the best estimate of the bid
        self.budget_spend += (cost / 1e9)

        # print(self.budget_spend)

        if cost > 0:
            self.wins_e += 1
            self.total_wins += 1
            self.total_rewards += reward

        action = min(self.target_value * state['click_prob'] * 1e9,
                        (self.budget - self.budget_spend) * 1e9)

        return action

    def done(self):
        return self.budget <= self.budget_spend

def main():
    # Instantiate the Environment and Agent
    env = gym.make('AuctionEmulator-v0')
    env.seed(0)
    agent = LinearBidAgent()
    obs, reward, cost, done = env.reset()
    agent.cur_day = obs['weekday']

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