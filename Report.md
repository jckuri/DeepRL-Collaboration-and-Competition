# DeepRL-Collaboration-and-Competition
Project 3 "Collaboration and Competition" of the Deep Reinforcement Learning nanodegree.

## Learning Algorithm

My `Project 2 - Continuous Control` <https://github.com/jckuri/DeepRL-Continuous-Control> was a total sucess because I solved the problem in just 168 episodes! Given that successful experience, I decided to reuse the code and hyperparameters of my DDPG Agent in order to build my new MADDPG Agent for Project 3.

Given the incremental nature of this project, the explanation of the learning algorithm is divided in 4 incremental parts:
1. Introduction to Deep Reinforcement Learning;
2. Details of my DDPG implementation;
3. How I modified the Udacity's DDPG Pendulum to make this project work so well;
4. How I implemented my MADDPG Agent.

### Part 1. Introduction to Deep Reinforcement Learning

I'm an expert in Deep Reinforcement Learning as you can see in the seminars I gave last year:

Webinar on Deep Reinforcement Learning (Secure & Private AI Scholarship Challenge)<br/>
https://youtu.be/oauLZG9nAX0

Deep Reinforcement Learning (IEEE Computer Society)<br/>
https://youtu.be/rf91xKkoP6w

In summary, Deep Reinforcement Learning (Deep RL) is an extension of Reinforcement Learning (RL) that uses Deep Learning techniques in order to handle complex continuous states instead of simple discrete states. RL is the branch of artificial intelligence that programs algorithms capable of learning and improving from experience in the form of SARS tuples: State 0, Action, Reward, and State 1. This kind of algorithms and problems are very general because they are based on search algorithms, general problem solvers (GPS), and statistics.

In my presentation at http://bit.do/DeepRL, there is a great explanation of the DQN algorithm:
- The DQN Agent senses the environment and take some actions to maximize the total reward.
- The deep neural network to represent complex states can be as complex as a convolutional neural network capable of seeing raw pixels; and actions can be the combinations of directions and buttons of a joystick.
- We define a loss function based on the formulae of RL. In that way, an RL problem is transformed into a supervised learning problem because we can apply the gradient descent technique to minimize the loss function.

<p align="center">
 <img src="/images/math.png">
</p>

### Part 2. Details of my DDPG implementation

The Project 2 goes beyond DQN. Because it includes new Deep RL techniques:
- **Actor-critic method** in which the actor computes policies to act and the critic helps to correct the policies based on its Q-values;
- **Deep Deterministic Policy Gradients (DDPG)**, which is similar to actor-critic methods but it differs because the actor produces a deterministic policy instead of stochastic policies; the critic evaluates such deterministic policy; and the actor is trained by using the deterministic policy gradient algorithm;
- **Two sets of Target and Local Networks**, which is a way to implement the double buffer technique in order to avoid oscillations caused by overestimated values;
- **Soft Updates** instead of hard updates so that the values of the local networks are slowly transferred to the target networks;
- **Replay Buffer** in order to keep training the DDPG Agent with past experiences;
- **Ornstein-Uhlenbeck(O-U) Noise** which is introduced at training in order to make the network learn in a more robust and more complete way.

Moreover, the DDPG Agent uses 2 deep neural networks to represent complex continuous states. 1 neural network for the actor and 1 neural network for the critic.

The neural network for the actor has:
- A linear fully-connected layer of dimensions state_size=`state_size` and fc1_units=128;
- The ReLu function;
- Batch normalization;
- A linear fully-connected layer of dimensions fc1_units=128 and fc2_units=128;
- The ReLu function;
- A linear fully-connected layer of dimensions fc2_units=128 and action_size=`action_size`;
- The tanh function.

The neural network for the critic has:
- A linear fully-connected layer of dimensions state_size=`state_size` and fcs1_units=128;
- The ReLu function;
- Batch normalization;
- A linear fully-connected layer of dimensions fcs1_units=128 + `action_size` and fc2_units=128;
- The ReLu function;
- A linear fully-connected layer of dimensions fc2_units=128 and output_size=1;

This implementation has the following metaparameters:

```
# replay buffer size (a very big database of SARS tuples)
BUFFER_SIZE = int(1e5)  

# minibatch size (the number of experience tuples per training iteration)
BATCH_SIZE = 128        

# discount factor (the Q-Network is aware of the intermediate future, but not the far future)
GAMMA = 0.99            

# for soft update of target parameters 
TAU = 1e-3           

# learning rate of the actor 
LR_ACTOR = 2e-4         

# learning rate of the critic 
LR_CRITIC = 2e-4        

# L2 weight decay
WEIGHT_DECAY = 0        
```

### Part 3. How I modified the Udacity's DDPG Pendulum to make this project work so well

I copied, pasted, and slightly modified the source code of the Udacity repository https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-pendulum.

First, Udacity's DDPG Pendulum uses OpenAI Gym and this project uses Unity to simulate the environment. Hence, I needed to modify the code in the Jupyter notebook to make it work properly.

Then, I changed the number of hidden units for the actor and the critic: `fc1_units=128, fc2_units=128`

```
    #def __init__(self, state_size, action_size, seed, fc1_units=400, fc2_units=300):
    def __init__(self, state_size, action_size, seed, fc1_units=128, fc2_units=128): # ADDED
```

```
    #def __init__(self, state_size, action_size, seed, fcs1_units=400, fc2_units=300):
    def __init__(self, state_size, action_size, seed, fcs1_units=128, fc2_units=128): # ADDED
```

I added batch normalization after the first layer of the actor:

```
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.bn1 = nn.BatchNorm1d(fc1_units) # ADDED
        self.reset_parameters()
```

```
    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        if state.dim() == 1: state = torch.unsqueeze(state,0) # ADDED
        x = F.relu(self.fc1(state))
        x = self.bn1(x) # ADDED
        x = F.relu(self.fc2(x))
        return F.tanh(self.fc3(x))
```

I added batch normalization after the first layer of the critic:

```
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fcs1 = nn.Linear(state_size, fcs1_units)
        self.fc2 = nn.Linear(fcs1_units+action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        self.bn1 = nn.BatchNorm1d(fcs1_units) # ADDED
        self.reset_parameters()
```

```
    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        if state.dim() == 1: state = torch.unsqueeze(state,0) # ADDED
        xs = F.relu(self.fcs1(state))
        xs = self.bn1(xs) # ADDED
        x = torch.cat((xs, action), dim=1)
        x = F.relu(self.fc2(x))
        return self.fc3(x)
```

I changed the learning rates of the actor and the critic to `2e-4`:

```
#LR_ACTOR = 1e-4         # learning rate of the actor
#LR_CRITIC = 1e-3        # learning rate of the critic
LR_ACTOR = 2e-4         # learning rate of the actor # ADDED
LR_CRITIC = 2e-4        # learning rate of the critic # ADDED
```

I copied the initial weights of the local networks to the target networks. In this way, local networks and target networks start with equal values. This action could improve the stability of the initial part of training.

```
        self.clone_weights(self.actor_target, self.actor_local) # ADDED
        self.clone_weights(self.critic_target, self.critic_local) # ADDED
```

```
    def clone_weights(self, w1, w0): # ADDED
        for p1, p0 in zip(w1.parameters(), w0.parameters()):
            p1.data.copy_(p0.data)
```

Moreover, I clipped the values of the critic to a maximum of `1`. This action was suggested by the video lecture in `Part 3. Policy-Based Methods; Lesson 4: Proximal Policy Optimization; 11. PPO Part 2: Clipping Policy Updates`.

```
        # ---------------------------- update critic ---------------------------- #
        ...
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1) # ADDED
```

In the class of Ornstein-Uhlenbeck noise, I decreased the value of sigma a little bit:

```
    #def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.1): # ADDED
```

Finally, I found a bug in the Udacity's DDPG Pendulum project. After correcting this bug, the learning curve improved dramatically. In brief, sigma should be multiplied by a normal distribution, and not by a random distribution with range `[0,1)`, which is counterintuitive. In the previous and wrong version, the sigma factor only contributed in a positive way, without negative values. But I think negative values are also important. So, I corrected this bug and the learning curve improved in a radical way.

```
    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        #dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x)) # ADDED
        self.state = x + dx
        return self.state
```

### Part 4. How I implemented my MADDPG Agent

First, I put each class in a separate file, resulting in more Python files: Tennis.ipynb, ddpg.py, maddpg.py, model.py, ounoise.py, and replay_buffer.py.

The replay buffer and the method step were deleted from the DDPG module. Now both concepts are centralized in the MADDPG class:

```
    def __init__(self, hp):
        self.hp = hp
        self.memory = replay_buffer.ReplayBuffer(hp)
        self.agents = [ddpg.Agent(self.hp) for _ in range(self.hp.num_agents)]
        self.losses = (0., 0.)
```

```

    def step(self, states, actions, rewards, next_states, dones):
        for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
            self.memory.add(state, action, reward, next_state, done)
        if len(self.memory) < self.hp.batch_size: return
        critic_losses = []
        actor_losses = []
        for agent in self.agents:
            experiences = self.memory.sample()
            critic_loss, actor_loss = agent.learn(experiences)
            critic_losses.append(critic_loss)
            actor_losses.append(actor_loss)
        self.losses = (np.mean(critic_losses), np.mean(actor_losses))
```

In the Jupyter notebook Tennis.ipynb, I collected all hyperparameters into 1 object and such object is passed to all the corresponding classes. And such classes needed some modifications to deal with that new object of hyperparameters. Previously, hyperparameters were passed as separate parameters through functions. Now all hyperparameters are passed in a single object.

```
hp = HyperParameters()

hp.num_agents = num_agents
hp.state_size = 24
hp.action_size = 2
hp.random_seed = 222
hp.buffer_size = int(1e5)  # replay buffer size
hp.batch_size = 128        # minibatch size
hp.gamma = 0.99            # discount factor
hp.tau = 1e-3              # for soft update of target parameters
hp.lr_actor = 1e-4 #2e-4         # learning rate of the actor # ADDED
hp.lr_critic = 1e-4 #2e-4        # learning rate of the critic # ADDED
hp.weight_decay = 0        # L2 weight decay
hp.print_every = 100
hp.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
```

Another important modification was the training and testing loops. In the single agent case, we had code like this `state = env_info.vector_observations[0]` that takes the first observation for an unique agent. Now we have code like this `states = env_info.vector_observations` that takes the whole list of observations. In general, all aspects now have 2 agents instead of just 1 agent.

```
def train_maddpg(n_episodes = 50000):
    scores = []
    scores_deque = deque(maxlen = 100)
    avg_scores = []
    for iteration in range(1, n_episodes + 1):
        env_info = env.reset(train_mode = True)[brain_name]
        states = env_info.vector_observations
        maddpg.reset()
        score = np.zeros(hp.num_agents)
        while True:
            actions = maddpg.act(states)
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done
            maddpg.step(states, actions, rewards, next_states, dones)
            score += rewards
            states = next_states
            if np.any(dones): break
        max_score = np.max(score)
        scores.append(max_score)
        scores_deque.append(max_score)
        avg_score = np.mean(scores_deque) 
        avg_scores.append(avg_score)
        print('\rEpisode {}\tAverage Score: {:.4f}'.format(iteration, avg_score), end="")
        if iteration % hp.print_every == 0:
            print('\rEpisode {}\tAverage Score: {:.4f}'.format(iteration, avg_score))
            save_weights()
        if avg_score >= 0.5:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.4f}'.format(iteration, avg_score))
            save_weights()
            break
    return scores, avg_scores

scores, avg_scores = train_maddpg()
```

After adapting all the code to make it work for multiple agents, the code worked perfectly and was solved (score >= 0.5) in the very first attempt. I was surprised. Then I changed both learning rates (actor and critic) to 1e-4. And it worked even beter.

## Plot of Rewards

The DDPG Agents were trained for `775` episodes. In each episode, the agents are trained from the begining to the end of the simulation. Some episodes are larger and some episodes are shorter, depending when the ending condition of each episode appears. Each episode has many iterations. In each iteration, the DDPG Agents are trained with `BATCH_SIZE=128` experience tuples (SARS).

```
Episode 100	Average Score: 0.0110
Episode 200	Average Score: 0.0060
Episode 300	Average Score: 0.0432
Episode 400	Average Score: 0.0523
Episode 500	Average Score: 0.0455
Episode 600	Average Score: 0.0664
Episode 700	Average Score: 0.1158
Episode 775	Average Score: 0.5023
Environment solved in 775 episodes!	Average Score: 0.5023
```

The rubric asks to obtain an average score of 0.5 or more for 100 episodes. The best model was saved. In the graph, the blue lines connect the scores in each episode. Whereas the red lines connect the average scores in each episode. **The problem was solved in just 775 episodes, which is awesome!**

![Plot of rewards (training)](/images/plot-of-rewards-training.png)

After training, the saved model was loaded and tested for 50 episodes. Here are the results of such testing. You can see that, on average, the scores are greater than 0.5. 

```
Episode 1	Score: 1.1000
Episode 2	Score: 2.6000
Episode 3	Score: 0.6000
Episode 4	Score: 0.1000
Episode 5	Score: 0.0000
Episode 6	Score: 1.0000
Episode 7	Score: 1.4000
Episode 8	Score: 0.6000
Episode 9	Score: 0.1000
Episode 10	Score: 0.4000
Episode 11	Score: 0.2000
Episode 12	Score: 0.3000
Episode 13	Score: 0.1000
Episode 14	Score: 1.4000
Episode 15	Score: 2.6000
Episode 16	Score: 2.1000
Episode 17	Score: 2.3900
Episode 18	Score: 0.3000
Episode 19	Score: 0.5000
Episode 20	Score: 0.2000
Episode 21	Score: 0.1000
Episode 22	Score: 0.4000
Episode 23	Score: 0.1000
Episode 24	Score: 2.3000
Episode 25	Score: 2.6000
Episode 26	Score: 2.7000
Episode 27	Score: 0.3000
Episode 28	Score: 2.6000
Episode 29	Score: 0.9900
Episode 30	Score: 1.6000
Episode 31	Score: 0.1000
Episode 32	Score: 1.7000
Episode 33	Score: 1.6900
Episode 34	Score: 1.9900
Episode 35	Score: 2.6000
Episode 36	Score: 1.3000
Episode 37	Score: 0.4000
Episode 38	Score: 1.7000
Episode 39	Score: 2.2000
Episode 40	Score: 1.3000
Episode 41	Score: 2.6000
Episode 42	Score: 1.8000
Episode 43	Score: 1.4000
Episode 44	Score: 1.8000
Episode 45	Score: 1.9000
Episode 46	Score: 2.6000
Episode 47	Score: 2.7000
Episode 48	Score: 2.6000
Episode 49	Score: 0.2000
Episode 50	Score: 2.6000
```

In the graph, the blue lines connect the scores in each episode. The red horizontal line represents the value of 0.5. Notice that the big majority of points are greater than the red horizontal line of 0.5.

![Plot of rewards (testing)](/images/plot-of-rewards-testing.png)

## Ideas for Future Work

Since MADDPG is based on multiple DDPG Agents, I will repeat the recommendations I wrote for my Project 2:
https://github.com/jckuri/DeepRL-Continuous-Control/blob/master/Report.md#ideas-for-future-work

I can improve this DDPG Agent because it is just a copy of the source code of the Udacity repository https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-pendulum. I even copied the hyperparameters and I modified them just a little bit. I did an excellent job at meta-optimizing the hyperparameters. However, I should accept that there could be a better set of hyperparameters. Therefore, there is room for improvement.

So far, this implementation has only `5` techniques: 
- Deep Deterministic Policy Gradients (DDPG);
- Two sets of Target and Local Networks;
- Soft Updates;
- Replay Buffer;
- Ornstein-Uhlenbeck(O-U) Noise.

Future implementations can be improved by applying the following techniques:
- **Prioritized** experience replay;
- Distributed learning with multiple independent agents (TRPO, PPO, A3C, and A2C);
- Q-prop algorithm, which combines both off-policy and on-policy learning.
