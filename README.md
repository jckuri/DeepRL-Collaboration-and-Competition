# DeepRL-Collaboration-and-Competition
Project 3 "Collaboration and Competition" of the Deep Reinforcement Learning nanodegree.

## Training Code

You can find the training code here: [Tennis.ipynb](Tennis.ipynb), [ddpg.py](ddpg.py), [maddpg.py](maddpg.py), [model.py](model.py), [ounoise.py](ounoise.py), and [replay_buffer.py](replay_buffer.py).

## Saved Model Weights

You can find the saved model weights here: [checkpoint_actor_0.pth](checkpoint_actor_0.pth), [checkpoint_actor_1.pth](checkpoint_actor_1.pth), [checkpoint_critic_0.pth](checkpoint_critic_0.pth), and [checkpoint_critic_1.pth](checkpoint_critic_1.pth).

## Project Details

For this project, you will work with the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment.

<p align="center">
 <img src="/images/collaborative-pingpong.gif">
</p>

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single score for each episode.

The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5. **LIKE THIS:**

![Plot of rewards (training)](/images/plot-of-rewards-training.png)

## Getting Started

Follow the instructions in this link in order to install all the dependencies required to run this project:<br/>
https://github.com/udacity/deep-reinforcement-learning#dependencies

Download the `Project 3 - Collaboration and Competition` into your computer:<br/>
https://github.com/jckuri/DeepRL-Collaboration-and-Competition

Follow the instructions in this link in order to install the Unity environment required to run this project:<br/>
https://github.com/udacity/deep-reinforcement-learning/tree/master/p3_collab-compet#getting-started

The easiest way to install the requirements is to use the file [requirements.txt](python/requirements.txt)
```
tensorflow==1.7.1
Pillow>=4.2.1
matplotlib
numpy>=1.11.0
jupyter
pytest>=3.2.2
docopt
pyyaml
protobuf==3.5.2
grpcio==1.11.0
torch==0.4.0
pandas
scipy
ipykernel
```

Execute this command in order to install the software specified in `requirements.txt`<br/>
```pip -q install ./python```<br/>
This command is executed at the beginning of the Jupyter notebook [Tennis.ipynb](Tennis.ipynb).

If you have troubles when installing this project, you can write me at:<br/>
https://www.linkedin.com/in/jckuri/

## Instructions

Follow the instructions in [Tennis.ipynb](Tennis.ipynb) to get started with training your own agents!

To run the Jupyter notebook, use the following Unix command inside the project's directory:

```
jupyter notebook Tennis.ipynb
```

To run all the cells in the Jupyter notebook again, go to the Jupyter notebook menu, and click on `Kernel` => `Restart & Run All`.

At the end of the Jupyter notebook, there is a space in which you can program your own implementation of this MADDPG Agent.

## Report

You can find the report here: [Report.md](Report.md)
