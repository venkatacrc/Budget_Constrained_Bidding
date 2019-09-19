"""
    Auction Emulator to generate bid requests from iPinYou DataSet.
"""
import gym
from gym import error, spaces, utils
from gym.utils import seeding

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
        fin = open(file_in, 'r')
        self._step = 0
        self.bids = []
        for bid in fin:
            bid_vals = bid.split(" ")
            self.bids.append(bid_vals) 
        self.num_episodes = len(self.bids)
        fin.close()


    def _get_bid_fields(self):
        self.mkt_price = bid_request[FLAGS.market_price]
        self.click = bid_request[FLAGS.click]
        self.conversion = bid_request[FLAGS.conversion]
 
    def reset(self):
        """
        Reset the OpenAI Gym Auction Emulator environment.
        """
        self._step = 0
        bid_request = self.bids[self._step]
        self._get_bid_fields()
        self._step += 1
        return bid_request

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
        bid_request = self.bids[self._step]
        self._get_bid_fields()
        self._step += 1
        done = self._step >= self.num_episodes

        return done, win_impression, user_click, user_conversion


    def render(self, mode='human'):
        pass

    def close(self):
        self.bids = [] 