"""
    Auction Emulator to generate bid requests from iPinYou DataSet.
"""
import gym
from gym import error, spaces, utils
from gym.utils import seeding

import configparser
import gflags

FLAGS = gflags.FLAGS

class AuctionEmulatorEnv(gym.Env):
    """
    AuctionEmulatorEnv can be used with Open AI Gym Env and is used to generate
    the bid requests reading the dataset files.
    """
    def __init__(self, file_in):
        """
        Args:
            file_in: Dataset input file
        Populates the bid requests to self.bids list.
        """
        self._step = 0
        with open(file_in, 'r') as f:
            self.bid_requests = [br.rstrip('\n').split('\t') for br in f]
        self.num_episodes = len(self.bid_requests)

    def _load_config(self):
        cfg = configparser.ConfigParser()
        cfg.read('./config.cfg')
        self.bid_request_fields = {}
        for item in cfg['bid_request']:
            self.bid_request_fields[item] = int(cfg['bid_request'][item])

    def _get_bid_fields(self):
        self.mkt_price = self.bid_request[FLAGS.market_price]
        self.click = self.bid_request[FLAGS.click]
        self.conversion = self.bid_request[FLAGS.conversion]
 
    def reset(self):
        """
        Reset the OpenAI Gym Auction Emulator environment.
        """
        self._step = 0
        self.bid_request = self.bids[self._step]
        self._get_bid_fields()
        self._step += 1
        return self.bid_request

    def step(self, action):
        """
        Args:
            action: bid response (bid_price)
        """
        bid_price = action #TODO: bid_price for now.
        if bid_price >= self.mkt_price:
            win_impression = 1
            user_click = self.click
            user_conversion = self.conversion
        self.bid_request = self.bids[self._step]
        self._get_bid_fields()
        self._step += 1
        done = self._step >= self.num_episodes

        return done, win_impression, user_click, user_conversion


    def render(self, mode='human'):
        pass

    def close(self):
        self.bids = []
