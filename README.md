# CSE571 Team Project: REINFORCE and Actor Critic with Softmax Policy

Matthew Groholski, Rodney Staggers, Anjali Mohanthy, Neha Shaik

**NOTE:** Project was developed using Python 3.6.15.

## About

REINFORCE is an algorithm that finds optimal feature weights using gradient descent with a differentiable stochastic policy. Read more at https://link.springer.com/article/10.1007/BF00992696.

## REINFORCE Agent
### Instructions 

To run the REINFORCE agent with default parameters, use `python pacman.py -p ReinforceAgent`.

### Parameters
`-n` : Specify the number of episodes to train on.

`-q` : Run in quiet mode.

`-g` : Specify ghost agent type.

`-k` : Specify the number of ghost agents.

`-n` : Specify the number of games to play.

`-l` : Specify the layout of the grid.

`-a` : Comma seperated values sent to the agent. For learning rate and reward discount,  `-a alpha=0.2,gamma=2`



## Actor Critic Agent
### Instructions
