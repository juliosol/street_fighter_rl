import retro
import gym
from gym import wrappers
from collections import defaultdict
import numpy as np
#from QLearning import * 

class QLearnAgent():
    """
    """
    
    def __init__(self, q_dict, gamma, alpha, epsilon, env):
        self.Q = q_dict
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.env = env

    ## Q values update equation
    ## Q(s,a) <- Q(s,a) + alpha * (reward + gamma * max_a Q(s,a) - Q(s,a))
    def update_Q(self, s, r, a, s_next, done, actions, Q, alpha=None, gamma=None):
        if alpha:
            self.alpha = alpha
        if gamma:
            self.gamma = gamma

        self.Q = Q
        
        max_q_next = max([self.Q[s_next, a] for a in actions])
        self.Q[s,a] += self.alpha * (r + \
                       self.gamma * max_q_next * (1.0 - done) - self.Q[s,a])

    def act(self, ob, actions, q_dict=None, epsilon=None):
        if epsilon:
            self.epsilon = epsilon
        if q_dict:
            self.Q = q_dict

        random_val = np.random.random()
        if random_val < self.epsilon:
            return self.env.action_space.sample()

        qvals = {a: self.Q[ob, a] for a in actions}
        max_q = max(qvals.values())

        # In case multiple actions have same maximum q value
        actions_with_max_q = [a for a, q in qvals.items() if q == max_q]
        return np.random.choice(actions_with_max_q)


class RetroActionDiscretizer(gym.ActionWrapper):
    """
    Wrap a gym-retro environment and make it 
    use discrete actions.
    """
    def __init__(self, env):
        super(RetroActionDiscretizer, self).__init__(env)
        buttons = ["B", "A", "MODE", "START", "UP", "DOWN", "LEFT",
                   "RIGHT", "C", "Y", "X", "Z"]
        actions = [["LEFT"], ["RIGHT"], ["UP"], ["DOWN"], ["DOWN", "LEFT"],
                   ["DOWN", "RIGHT"], ["UP","LEFT"], ["UP","RIGHT"],
                   ["DOWN", "Y"], ["DOWN", "X"], ["DOWN", "Z"],
                   ["DOWN", "B"], ["DOWN","A"], ["UP", "Y"], 
                   ["UP", "X"], ["UP", "Z"], ["UP", "B"], 
                   ["UP","A"]]
        self._actions = []
        for action in actions:
            arr = np.array([False]*12)
            for button in action:
                arr[buttons.index(button)] = True
            self._actions.append(arr)
        self.action_space = gym.spaces.Discrete(len(self._actions))

    def action(self, a):
        return self._actions[a].copy()

    #def sample(self, a)


class RetroDiscretizedObservationWrapper(gym.ObservationWrapper):
    """
    Wrapper converts a Box observation into a single integer.
    """
    def __init__(self, env, n_bins=10, low=None, high=None):
        super().__init__(env)
        #assert isinstance(env.observation_space,)

        low = self.observation_space.low if low is None else low
        high = self.observation_space.high if high is None else high

        self.n_bins = n_bins
        self.val_bins = [np.linspace(l, h, n_bins + 1) for l, h in 
                         zip(low.flatten(), high.flatten())]
        self.observation_space = gym.spaces.Discrete(n_bins ** low.flatten().shape[0])

    def _convert_to_one_number(self, digits):
        return sum([d * ((self.n_bins + 1) ** i) for i, d in enumerate(digits)])

    def observation(self, observation):
        digits = [np.digitize([x], bins)[0]
                  for x, bins in zip(observation.flatten(), self.val_bins)]

        return self._convert_to_one_number(digits)
        

def main():

    Q = defaultdict(float)
    random_Q = defaultdict(float)
    gamma = 0.9 # Discounting factor
    alpha = 0.2 # Updating parameter
    n_steps = 1000000
    epsilon = 0.01 # 0.1% chances of applying a random action

    #env = retro.make(game='StreetFighterIISpecialChampionEdition-Genesis')
    env = gym.make('Breakout-ram-v0')

    import pdb
    pdb.set_trace()

    env = gym.wrappers.Monitor(env, 'rec_1')
    
    if not isinstance(env.observation_space, gym.spaces.discrete.Discrete):
        discr_env = RetroDiscretizedObservationWrapper(env, n_bins=8, low=np.array([-2.4, -2.0, -0.42, -3.5]),
                                              high=np.array([2.4, 2.0, 0.42, 3.5]))

    if not isinstance(discr_env.action_space, gym.spaces.discrete.Discrete):
        ssf_env = RetroActionDiscretizer(discr_env)
    
    ssf_env = discr_env

    actions = range(env.action_space.n)    
    
    ob = ssf_env.reset()

    q_agent = QLearnAgent(Q, gamma, alpha, epsilon, ssf_env)
    
    rewards = []
    reward = 0.0 

    for step in range(n_steps):
        print("This is step {}".format(step))
        
        a = q_agent.act(ob, actions)
        
        ob_next, r, done, _ = ssf_env.step(a)
        q_agent.update_Q(ob, r, a, ob_next, done, actions, q_agent.Q)
        reward += r
        #print("This is reward ", reward)
        if done:
            print("Reward after done is ", reward)
            rewards.append(reward)
            reward = 0.0
            ob = ssf_env.reset()
        else:
            ob = ob_next
    ssf_env.close()
    
    """
    print("Starting random trainings")
    #env = retro.make(game='StreetFighterIISpecialChampionEdition-Genesis')
    env = gym.make('Breakout-ram-v0')

    env = gym.wrappers.Monitor(env, 'breakout_random_recording_v2')
    
    if not isinstance(env.observation_space, gym.spaces.discrete.Discrete):
        discr_env = RetroDiscretizedObservationWrapper(env, n_bins=8, low=np.array([-2.4, -2.0, -0.42, -3.5]),
                                              high=np.array([2.4, 2.0, 0.42, 3.5]))

    if not isinstance(discr_env.action_space, gym.spaces.discrete.Discrete):
        ssf_env = RetroActionDiscretizer(discr_env)
    
    ssf_env = discr_env

    actions = range(env.action_space.n)

    random_agent = QLearnAgent(random_Q, gamma, alpha, 1, ssf_env)

    random_ob = ssf_env.reset()
    random_rewards = []
    random_reward = 0.0
    for step in range(n_steps):
        print("This is step {}".format(step))
        random_a = random_agent.act(random_ob, actions)
        random_ob_next, random_r, done, _ = ssf_env.step(random_a)
        random_agent.update_Q(random_ob, random_r, random_a,
                              random_ob_next, done, actions, random_agent.Q)
        random_reward += random_r
        if done:
            random_rewards.append(random_reward)
            random_reward = 0.0
            random_ob = ssf_env.reset()
        else:
            random_ob = random_ob_next
    ssf_env.close()
    """
    
if __name__ == "__main__":
    main()