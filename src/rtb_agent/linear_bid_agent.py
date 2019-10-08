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

env = gym.make('AuctionEmulator-v0')

# Reset the environment
state, reward, done = env.reset()
# print(state)

while not done:
    # action = bid amount
    action = env.act(state)
    next_state, reward, done = env.step(action)
    state = next_state # Next state assigned to current state

print("Total Clicks won with Budget={} amount = {}".format(env.budget, env.clicks_won))