# CSE571 Team Project: REINFORCE and Actor Critic with Softmax Policy

Matthew Groholski, Rodney Staggers, Anjali Mohanthy, Neha Shaik

**NOTE:** Project was developed using Python 3.6.15.

## About

REINFORCE is an algorithm that finds optimal feature weights using gradient descent with a differentiable stochastic policy. Read more at https://link.springer.com/article/10.1007/BF00992696.

Actor Critic is an algorithm that determines a policy through an actor that executes the current policy with a set of feature weights and a critic the gives an expected value for the current state. Read more at https://dilithjay.com/blog/actor-critic-methods#the-actor-critic-algorithm.

## REINFORCE Agent

### Instructions

To run the REINFORCE agent with default parameters, use `python pacman.py -p ReinforceAgent`.

### Parameters

`-n` : Specify the number of episodes to run on.

`-x` : Specify the number of episodes that are training.

`-q` : Run in quiet mode.

`-g` : Specify ghost agent type.

`-k` : Specify the number of ghost agents.

`-n` : Specify the number of games to play.

`-l` : Specify the layout of the grid.

`-a` : Comma seperated values sent to the agent. For learning rate and reward discount, `-a alpha=0.2,gamma=2`

**To reproduce results:** `python pacman.py -p ReinforceAgent -n 80 -x 60 -a gamma=0.8,alpha=0.2 -q`

## Actor Critic Agent

### Instructions

To run the REINFORCE agent with default parameters, use `python pacman.py -p ActorCriticAgent`.

`-n` : Specify the number of episodes to run on.

`-x` : Specify the number of episodes that are training.

`-q` : Run in quiet mode.

`-g` : Specify ghost agent type.

`-k` : Specify the number of ghost agents.

`-n` : Specify the number of games to play.

`-l` : Specify the layout of the grid.

`-a` : Comma seperated values sent to the agent. For learning rate and reward discount, `-a alpha=0.2,gamma=2`

**To reproduce results:** `python pacman.py -p ActorCriticAgent -n 80 -x 60 -a alpha_theta=0.25,alpha_w=0.15,gamma=0.9 -q`
