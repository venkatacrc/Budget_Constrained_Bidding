"""
    Auction Emulator to generate bid requests from iPinYou DataSet.
"""
import gym
from gym import error, spaces, utils
from gym.utils import seeding

import configparser
import os

class AuctionEmulatorEnv(gym.Env):
    """
    AuctionEmulatorEnv can be used with Open AI Gym Env and is used to generate
    the bid requests reading the iPinYou dataset files.
    Toy data set with 100 lines are included in the data directory.
    """
    metadata = {'render.modes': ['human']}

    def _load_config(self):
        """
        Parse the config.cfg file
        """
        cfg = configparser.ConfigParser()
        env_dir = os.path.dirname(__file__)
        cfg.read(env_dir + '/config.cfg')
        # print(cfg['data']['dtype'])
        if cfg['data']['dtype'] == 'ipinyou':
            self.file_in = env_dir + str(cfg['data']['ipinyou_path'])
        self.idx_click = int(cfg['data']['idx_click'])
        self.idx_bidprice = int(cfg['data']['idx_bidprice'])
        self.idx_payprice = int(cfg['data']['idx_payprice'])

    def __init__(self):
        """
        Args:
        Populates the bid requests to self.bid_requests list.
        """
        self._load_config()
        self._step = 1
        with open(self.file_in, 'r') as f:
            self.bid_requests = [br.rstrip('\n').split('\t') for br in f.readlines()]
        self.num_bids = len(self.bid_requests)
        self.bidprice = 0
        self.payprice = 0
 
    def reset(self):
        """
        Reset the OpenAI Gym Auction Emulator environment.
        """
        self._step = 1
        return self.bid_requests[self._step], 0, False

    def step(self, action):
        """
        Args:
            action: bid response (bid_price)
        Reward is computed using the bidprice to payprice difference.
        """
        done = False
        state = None
        r = 0
        self.bidprice = int(self.bid_requests[self._step][self.idx_bidprice])
        self.payprice = int(self.bid_requests[self._step][self.idx_payprice])
        self.click = int(self.bid_requests[self._step][self.idx_click])
        if self._step < self.num_bids - 1:
            state =  self.bid_requests[self._step]
            if action:
                r = self.click                   
        else:
            done = True
        self._step += 1
        return state, r, done

    def render(self, mode='human', close=False):
        pass

    def close(self):
        pass




