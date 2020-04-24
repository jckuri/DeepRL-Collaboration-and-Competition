# DeepRL-Continuous-Control
Project 2 "Continuous Control" of the Deep Reinforcement Learning nanodegree.

## Training Code

You can find the training code here: [Continuous_Control.ipynb](Continuous_Control.ipynb), [ddpg_agent.py](ddpg_agent.py), and [model.py](model.py).

## Saved Model Weights

You can find the saved model weights here: [checkpoint_actor.pth](checkpoint_actor.pth) and [checkpoint_critic.pth](checkpoint_critic.pth).

## Project Details

For this project, you will work with the [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment.

<p align="center">
 <img src="/images/reacher.gif">
</p>

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

**Option 1: Solve the First Version**

The task is episodic, and in order to solve the environment, your agent must get an average score of +30 over 100 consecutive episodes. **LIKE THIS:**

![Plot of rewards (training)](/images/plot-of-rewards-training.png)

## Getting Started

Follow the instructions in this link in order to install all the dependencies required to run this project:<br/>
https://github.com/udacity/deep-reinforcement-learning#dependencies

Download the `Project 2 - Continuous Control` into your computer:<br/>
https://github.com/jckuri/DeepRL-Continuous-Control

Follow the instructions in this link in order to install the Unity environment required to run this project:<br/>
https://github.com/udacity/deep-reinforcement-learning/tree/master/p2_continuous-control#getting-started

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
This command is executed at the beginning of the Jupyter notebook [Continuous_Control.ipynb](Continuous_Control.ipynb).

If you have troubles when installing this project, you can write me at:<br/>
https://www.linkedin.com/in/jckuri/

## Instructions

Follow the instructions in [Continuous_Control.ipynb](Continuous_Control.ipynb) to get started with training your own agent!

To run the Jupyter notebook, use the following Unix command inside the project's directory:

```
jupyter notebook Continuous_Control.ipynb
```

To run all the cells in the Jupyter notebook again, go to the Jupyter notebook menu, and click on `Kernel` => `Restart & Run All`.

At the end of the Jupyter notebook, there is a space in which you can program your own implementation of this DDPG Agent.

## Report

You can find the report here: [Report.md](Report.md)
