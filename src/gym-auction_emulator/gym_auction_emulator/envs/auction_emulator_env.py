"""
    Auction Emulator to generate bid requests from iPinYou DataSet.
"""
import gym
from gym import error, spaces, utils
from gym.utils import seeding

import configparser
import json
import os
import pandas as pd

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
        cfg = configparser.ConfigParser(allow_no_value=True)
        env_dir = os.path.dirname(__file__)
        cfg.read(env_dir + '/config.cfg')
        self.data_src = cfg['data']['dtype']
        if self.data_src == 'ipinyou':
            self.file_in = env_dir + str(cfg['data']['ipinyou_path'])
        self.metric = str(cfg['data']['metric'])

    def __init__(self):
        """
        Args:
        Populates the bid requests to self.bid_requests list.
        """
        self._load_config()
        self._step = 1
        fields =    [
                    'weekday',
                    'hour',
                    'auction_type',
                    'bidprice',
                    'slotprice',
                    'payprice',
                    'click_prob'
                    ]
        self.bid_requests = pd.read_csv(self.file_in, sep="\t", usecols=fields)
        self.total_bids = len(self.bid_requests)
        self.bid_line = {}

    def _get_observation(self, bid_req):
        observation = {}
        if bid_req is not None:
            observation['weekday'] = bid_req['weekday']
            observation['hour'] = bid_req['hour']
            observation['auction_type'] = bid_req['auction_type']
            observation['slotprice'] = bid_req['slotprice']
            observation['click_prob'] = bid_req['click_prob']
        return observation

    def _bid_state(self, bid_req):
        self.auction_type = bid_req['auction_type']
        self.bidprice = bid_req['bidprice']
        self.payprice = bid_req['payprice']
        self.click_prob = bid_req['click_prob']
        self.slotprice = bid_req['slotprice']

    def reset(self):
        """
        Reset the OpenAI Gym Auction Emulator environment.
        """
        self._step = 1
        bid_req = self.bid_requests.iloc[self._step]
        self._bid_state(bid_req)
        # observation, reward, cost, done
        return self._get_observation(bid_req), 0.0, 0.0, False

    def step(self, action):
        """
        Args:
            action: bid response (bid_price)
        Reward is computed using the bidprice to payprice difference.
        """
        done = False
        r = 0.0 # immediate reward
        r_p = 0.0 # temp reward
        c = 0.0 # cost for the bid impression

        if self.metric == 'clicks':
            r_p = self.click_prob
        else:
            raise ValueError(f"Invalid metric type: {self.metric}")

        mkt_price = max(self.slotprice, self.payprice)
        if action > mkt_price:
            if self.auction_type == 'SECOND_PRICE':
                r = r_p
                c = mkt_price
            elif self.auction_type == 'FIRST_PRICE':
                r = r_p
                c = action
            else:
                raise ValueError(f"Invalid auction type: {self.auction_type}")

        next_bid = None
        if self._step < self.total_bids - 1:
            next_bid = self.bid_requests.iloc[self._step]
            self._bid_state(next_bid)
        else:
            done = True

        self._step += 1

        return self._get_observation(next_bid), r, c, done

    def render(self, mode='human', close=False):
        pass

    def close(self):
        pass
