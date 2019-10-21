# Ad Tech Budget Constrained Real-Time Bidding (RTB)
This repo contains the Budget Constrained Real-Time Bidding Open AI Gym environment and Agents for Display Advertising Technology (Ad Tech). There are two agents trained for comparison. One is a Liner Bidding Agent and another one is a Model-free Reinforcement Learning(RL) based agent. The model-free RL agent is based on the paper from Wu et al. I have included the iPinYou and Fake dataset to run the models.

## Ad Exchange Open AI Gym environment

Open AI Gym environment mimics the Ad Exchange by taking the bid requests from the iPinYou or any other Bidding dataset. The agents can interact with it using the standard Gym API interface.

### Configuration parameters

Set the `dataset_path` to the absolute data set path location.
Set `metrics` to impressions, views, clicks, installs, conversions etc. 


## Reinforcement Learning(RL) Agent
The RL agent models the state by spilitting the entire day (24 Hours) into time steps, T, typically 15 minutes.


### Configuration paraemters
Set `T` to the time step duration.
Set `ACTION_SIZE` to the number of discrete positive and negative steps to control the bid impression price. 
[Here](http://bit.ly/AdTechRTB) are the slides for RTB project.

---

## How to Use:
Use the following command to run the Linear Bidding Agent.

```bash scripts/run_lin_bid.sh```

Use the following command to run the RL Bidding Agent.

```bash scripts/run_rl_bid.sh```

## Repository Citations

@misc{wu2018budget,
  title={Budget constrained bidding by model-free reinforcement learning in display advertising},
  author={Wu, Di and Chen, Xiujun and Yang, Xun and Wang, Hao and Tan, Qing and Zhang, Xiaoxun and Xu, Jian and Gai, Kun},
  booktitle={Proceedings of the 27th ACM International Conference on Information and Knowledge Management},
  pages={1443--1451},
  year={2018},
  organization={ACM}
}
