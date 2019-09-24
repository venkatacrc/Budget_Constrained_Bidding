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
            self.file_in = env_dir + '/../../../../data/ipinyou/data.txt'
        self.bidprice = int(cfg['data']['bidprice'])
        self.payprice = int(cfg['data']['payprice'])

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
        if self._step < self.num_bids - 1:
            state =  self.bid_requests[self._step]
            if action:
                r = int(self.bid_requests[self._step][self.bidprice]) - \
                    int(self.bid_requests[self._step][self.payprice])
        else:
            done = True
        self._step += 1
        return state, r, done

    def render(self, mode='human', close=False):
        pass

    def close(self):
        pass




