"""
Run this module inside Budget_Constrained_Bidding directory
%python3 src/rtb_agent/constant_bid_agent.py
"""
import sys,os
sys.path.append(os.getcwd()+'/src/gym-auction_emulator')
import gym, gym_auction_emulator

"""
Simple toy constant bidding agent that constantly bids $1 until budget runs out
This is an example to show the OpenAI gym interface for the 
    Budget Constrained Bidding problem.
"""

# initial budget
budget = 100

env = gym.make('AuctionEmulator-v0')

state, done = env.reset(), False

while not done:
    # action = bid amount
    if budget:
        action = env.target_price * env.
    else:
        action = 0
    budget -= env.payprice
    next_state, reward, done = env.step(action)
    if not done:
        print(next_state, reward, done)
