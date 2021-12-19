import tensorflow as tf
from tensorflow import keras
import numpy as np
#import retro
import gym
from gym import wrappers
from collections import defaultdict, deque
import cv2
from baselines.common.atari_wrappers import make_atari, wrap_deepmind, MaxAndSkipEnv, LazyFrames

 
class DQN(tf.keras.Model):
    """
    Class for the Q state-value function non-linear 
    approximator.
    """
    def __init__(self, num_actions):
        super(DQN, self).__init__()
        self.num_actions = num_actions
        self.conv1 = tf.keras.layers.Conv2D(16, 8, strides=4,
                                            activation='relu')
        self.conv2 = tf.keras.layers.Conv2D(32, 4, strides=2,
                                            activation='relu')

        self.flatten = tf.keras.layers.Flatten()        

        self.linear1 = tf.keras.layers.Dense(256, activation='relu')
        #print("This is num_actions " + str(self.num_actions))
        self.linear2 = tf.keras.layers.Dense(self.num_actions, activation='linear')

    def call(self, inputs, training=False):
        #import pdb
        #pdb.set_trace()
        output = self.conv1(inputs)
        output = self.conv2(output)
        output = self.flatten(output)
        output = self.linear1(output)
        output = self.linear2(output)
        return output


class DeepQLearner():
    """
    Agent using DQN for learning.
    """
    def __init__(self, num_actions, epsilon, gamma, env, optimizer):
        self.num_actions = num_actions
        self.epsilon = epsilon
        self.gamma = gamma
        self.env = env
        self.optimizer = optimizer
        #self.experience_replay_buffer = []

        self.q_network = DQN(self.num_actions)
        self.q_target = DQN(self.num_actions)
        self.copy_weights()

    def store(self, state, action, reward, next_state, terminated):
        self.experience_replay.append((state, action, reward, next_state, terminated))

    def copy_weights(self):
        self.q_target.set_weights(self.q_network.get_weights())

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return self.env.action_space.sample()
        
        #import pdb
        #pdb.set_trace()
        q_values = self.q_network.predict(state)

        return np.argmax(q_values)

class FrameResizing(gym.ObservationWrapper):
    def __init__(self, env, width=84, height=84, grayscale=True):
        """
        Wrapper that resizes a frame from the original size to an 
        (84, 84) frame following the Deepmind paper.
        Follows implementation of WarpFrame from baselines.common.atari_wrappers.
        """
        super().__init__(env)
        self._width = width
        self._height = height
        self._grayscale = grayscale
        if self._grayscale:
            num_colors = 1
        else:
            num_colors = 3

        new_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self._width, self._height, num_colors),
            dtype=np.uint8
        )

        original_space = self.observation_space
        self.observation_space = new_space
        assert original_space.dtype == np.uint8 and len(original_space.shape) == 3
        
    def observation(self, obs):
        frame = obs
        if self._grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.resize(frame, (self._width, self._height), interpolation=cv2.INTER_AREA)
        if self._grayscale:
            frame = np.expand_dims(frame, -1).astype(np.float32)
        obs = frame
        return obs


class StackFrames(gym.Wrapper):
    def __init__(self, env, k):
        """
        Stack k last frames.
        Follows implementation of FrameStack from baselines.common.atari_wrappers
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shape = env.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(*shape[:-1], shape[-1] * k),
                                                dtype=env.observation_space.dtype)

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, ac):
        ob, reward, done, info = self.env.step(ac)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))

class MaxAndSkipFrame(gym.Wrapper):
    def __init__(self, env, skip=4):
        """
        Taking maximum value for each pixel olour in current frame and previous
        frame and skips frames.
        Follows implementation of MaxAndSkipEnv from baselines.common.atari_wrappers
        """
        gym.Wrapper.__init__(self, env)
        self.skip = skip
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)

    def step(self, ac):
        """
        Repeat action, sum reward and choose maximum between
        current and last observation.
        """
        total_reward = 0.0
        done = None
        for i in range(self.skip):
            obs, reward, done, info = self.env.step(ac)
            if i == self.skip - 2:
                self._obs_buffer[0] = obs
                self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        max_frame = self._obs_buffer.max(axis=0)
        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class ClipRewards(gym.RewardWrapper):
    """
    Wrapper that will clip the reward of the environment to a given value.
    This particular implementation clips rewards to a value
    in {-1, 0, 1}
    Follows implementation of ClipRewardEnv in baselines.common.atari_wrappers.
    """
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
    
    def reward(self, reward):
        """
        Classify reward according to its sign. 
        """
        return np.sign(reward)

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
    seed = 230
    gamma = 0.8 # Discounting factor
    alpha = 0.3 # Updating parameter
    epsilon =0.01 # Epsilon greedy parameter
    epsilon_min = 0.1 # Minimum epsilon greedy parameter
    epsilon_max = 1.0 # Maximum epsilon greedy parameter
    epsilon_interval = (epsilon_max - epsilon_min) # Rate at which to reduce chance of random action being taken
    batch_size = 32
    max_steps_per_episode = 10000

    #sf_env = retro.make(game='StreetFighterIISpecialChampionEdition-Genesis')
    env = gym.make('BreakoutNoFrameskip-v4')
    env = StackFrames(ClipRewards(MaxAndSkipFrame(FrameResizing(env))), 4)
    env = gym.wrappers.Monitor(env, 'dqn_1')

    num_actions = env.action_space.n

    #Using Adam optimizer instead of RMSProp
    optimizer = keras.optimizers.Adam(learning_rate=0.0005, clipnorm=1.0)

    # Agent model used to make an action prediction used to make an action
    # The learner object has the prediction and target network.
    q_learner = DeepQLearner(num_actions, epsilon, gamma,
                             env, optimizer)

    #experience_replay = []
    # Experience replay buffers
    action_history = []
    state_history = []
    state_next_history = []
    rewards_history = []
    done_history = []
    episode_reward_history = []
    running_reward = 0
    episode_count = 0
    frame_count = 0

    # Number of frames to take random action to explore
    epsilon_random_frames = 50000
    #epsilon_random_frames = 0
    # Number of frames for exploration
    epsilon_greedy_frames = 1000000.0
    # Replay memory length
    max_replay_length = 100000
    # Train the model after 4 actions
    update_after_actions = 4
    # How often to update target network
    update_target_network = 100000
    # loss function
    loss_fn = keras.losses.Huber()

    while True: # Run until the algorithm is solved
        state = np.array(env.reset())
        episode_reward = 0

        for timestep in range(1, max_steps_per_episode):
            #env.render()
            # count frame count 
            frame_count += 1

            # Use epsilon-greedy for exploration
            if frame_count < epsilon_random_frames or epsilon > np.random.rand(1)[0]:
                # Take random action
                action = np.random.choice(num_actions)
            else:
                # Use model network to predict actions for Q values
                state_group = tf.convert_to_tensor(state)
                state_group = tf.expand_dims(state_group, 0)
                action_probs = q_learner.q_network(state_group)
                # Take best action
                action = tf.math.argmax(action_probs, 1)[0]

            # Reduce probability of taking random action
            epsilon -= epsilon_interval / epsilon_greedy_frames
            epsilon = max(epsilon, epsilon_min)

            # Taking the action selected and update values
            #import pdb
            #pdb.set_trace()
            state_next, reward, done, info = env.step(action)
            state_next = np.array(state_next)
            episode_reward += reward

            # Store information in replay memory
            #experience_replay.append((state, action, reward, state_next))
            action_history.append(action)
            state_history.append(state)
            state_next_history.append(state_next)
            done_history.append(done)
            rewards_history.append(episode_reward)
            state = state_next

            # Update every fourth frame and once batch size is over 32
            if frame_count % update_after_actions == 0 and len(done_history) > batch_size:

                print("Updating fourth frame.")
                # Get indices of samples for replay buffer
                indices = np.random.choice(range(len(done_history)), size=batch_size)

                # Using list compreseion to sample from replay buffer
                state_sample = np.array([state_history[i] for i in indices])
                state_next_sample = np.array([state_next_history[i] for i in indices])
                rewards_sample = [rewards_history[i] for i in indices]
                action_sample = [action_history[i] for i in indices]
                done_sample = tf.convert_to_tensor([float(done_history[i]) for i in indices])

                # Build updated Q-values for the sampled future states
                # Use target model for stability
                future_rewards = q_learner.q_target.predict(state_next_sample)
                # Q value = reward + discount factor * expected future reward
                updated_q_values = rewards_sample + gamma * tf.reduce_max(future_rewards, axis=1)

                # If final frame set the last value to -1
                updated_q_values = updated_q_values * (1 - done_sample) - done_sample

                # Create a mask so we only calculate loss on updated Q_values
                masks = tf.one_hot(action_sample, num_actions)

                with tf.GradientTape() as tape:
                    # Train the model on the states and updated Q-values
                    if state_sample.dtype != 'float32':
                        state_sample = state_sample.astype('float32')
                    q_values = q_learner.q_network(state_sample)

                    # Apply the masks to the Q-values to get the Q-values for the action taken
                    q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
                    # Calculate loss between new Q-value and old Q-value
                    loss = loss_fn(updated_q_values, q_action)

                # Backpropagation
                grads = tape.gradient(loss, q_learner.q_network.trainable_variables)
                optimizer.apply_gradients(zip(grads, q_learner.q_network.trainable_variables))

            if frame_count % update_target_network == 0:
                # update the target network with new weights
                q_learner.q_target.set_weights(q_learner.q_network.get_weights())
                # Log details
                template = "running reward: {:.2f} at episode {}, frame count {}"
                print(template.formate(running_reward, episode_count, frame_count))

            # Limit the state and reward history
            if len(rewards_history) > max_replay_length:
                del rewards_history[:1]
                del state_history[:1]
                del state_next_history[:1]
                del action_history[:1]
                del done_history[:1]

            if done:
                break

        # Update running reward to check condition for solving
        episode_reward_history.append(episode_reward)
        if len(episode_reward_history) > 100:
            del episode_reward_history[:1]
        running_reward = np.mean(episode_reward_history)

        episode_count += 1

        if running_reward > 40:  # Condition to consider the task solved
            print("Solved at episode {}!".format(episode_count))
            break
        #import pdb
        #pdb.set_trace()


if __name__ == "__main__":
    main()

