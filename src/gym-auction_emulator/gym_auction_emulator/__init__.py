from gym.envs.registration import register

register(
    id='AuctionEmulator-v0',
    entry_point='gym_auction_emulator.envs:AuctionEmulatorEnv',
)
