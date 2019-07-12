import threading
import numpy as np
import tensorflow as tf
import pylab
import time
import gym
from keras.layers import Dense, Input
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K
import pypownet.environment
import os
import pypownet.chronic


# global variables for threading
episode = 0
scores = []
time_step_end = 1000 # this is the maximum timesteps per episode
EPISODES_train = 500 # increase these episodes to 3k in the beginning as the episodes will be very short in the beginning
training_batch_size = 100 # memory size to be used for updating the NN model weights
num_threads = 1 # 48 or 16 or 32 - corresponds to parallel agents
hidden_layer_1_size = 200 # this is the shared layer between actor and critic
hidden_layer_2_size = 50 # this is the hidden layer size in actor and critic

input_dir = 'public_data/'
# game_over_mode = ["easy", "soft", "hard"]
# game_level = ["datasets_sub_7","datasets_sub_4","datasets"]
def set_environement(game_level = "datasets", start_id=40):
    """
        Load the first chronic (scenario) in the directory public_data/datasets
    """
    return pypownet.environment.RunEnv(parameters_folder=os.path.abspath(input_dir),
                                              game_level=game_level,
                                              chronic_looping_mode='natural', start_id=start_id,
                                              game_over_mode="easy")
    # return pypownet.environment.RunEnv(parameters_folder=os.path.abspath(input_dir),
    #                                           game_level=game_level,
    #                                           chronic_looping_mode='random', start_id=start_id,
    #                                           game_over_mode="hard")
    # return pypownet.environment.RunEnv(parameters_folder=os.path.abspath(input_dir),
    #                                           game_level=game_level,
    #                                           chronic_looping_mode='fixed', start_id=start_id,
    #                                           game_over_mode="hard")

# This is A3C(Asynchronous Advantage Actor Critic) agent(global)
class A3CAgent:
    def __init__(self, state_size, action_size, env_name):
        # get size of state and action
        self.state_size = state_size
        self.action_size = action_size

        # get gym environment name
        self.env_name = env_name

        # these are hyper parameters for the A3C
        self.actor_lr = 0.0001
        self.critic_lr = 0.0001
        self.discount_factor = .95
        self.hidden1, self.hidden2 = 200, 50
        self.threads = num_threads # 48 or 16 or 32 - corresponds to parallel agents

        # create model for actor and critic network
        self.actor, self.critic = self.build_model()

        # method for training actor and critic network
        self.optimizer = [self.actor_optimizer(), self.critic_optimizer()]

        self.sess = tf.InteractiveSession()
        K.set_session(self.sess)
        self.sess.run(tf.global_variables_initializer())

    # approximate policy and value using Neural Network
    # actor -> state is input and probability of each action is output of network
    # critic -> state is input and value of state is output of network
    # actor and critic network share first hidden layer
    def build_model(self):
        state = Input(batch_shape=(None,  self.state_size))
        shared = Dense(self.hidden1, input_dim=self.state_size, activation='relu', kernel_initializer='he_uniform')(state)

        actor_hidden = Dense(self.hidden2, activation='relu', kernel_initializer='he_uniform')(shared)
        action_prob = Dense(self.action_size, activation='softmax', kernel_initializer='he_uniform')(actor_hidden)

        value_hidden = Dense(self.hidden2, activation='relu', kernel_initializer='he_uniform')(shared)
        state_value = Dense(1, activation='linear', kernel_initializer='he_uniform')(value_hidden)

        actor = Model(inputs=state, outputs=action_prob)
        critic = Model(inputs=state, outputs=state_value)

        actor._make_predict_function()
        critic._make_predict_function()

        actor.summary()
        critic.summary()

        return actor, critic

    # make loss function for Policy Gradient
    # [log(action probability) * advantages] will be input for the back prop
    # we add entropy of action probability to loss
    def actor_optimizer(self):
        action = K.placeholder(shape=(None, self.action_size))
        advantages = K.placeholder(shape=(None, ))

        policy = self.actor.output

        good_prob = K.sum(action * policy, axis=1)
        eligibility = K.log(good_prob + 1e-10) * K.stop_gradient(advantages)
        loss = -K.sum(eligibility)

        entropy = K.sum(policy * K.log(policy + 1e-10), axis=1)

        actor_loss = loss + 0.01*entropy

        optimizer = Adam(lr=self.actor_lr)
        updates = optimizer.get_updates(self.actor.trainable_weights, [], actor_loss)
        train = K.function([self.actor.input, action, advantages], [], updates=updates)
        return train

    # make loss function for Value approximation
    def critic_optimizer(self):
        discounted_reward = K.placeholder(shape=(None, ))

        value = self.critic.output

        loss = K.mean(K.square(discounted_reward - value))

        optimizer = Adam(lr=self.critic_lr)
        updates = optimizer.get_updates(self.critic.trainable_weights, [], loss)
        train = K.function([self.critic.input, discounted_reward], [], updates=updates)
        return train

    # make agents(local) and start training
    def train(self):

        try:
            self.load_model('pypow_14_a3c')
            print("Loaded saved NN model parameters \n")
        except:
            print("No existing model is found or saved model sizes do not match - initializing random NN weights \n")
        agents = [Agent(i, self.actor, self.critic, self.optimizer, self.env_name, self.discount_factor,
                        self.action_size, self.state_size) for i in range(self.threads)]

        for agent in agents:
            agent.start()

        while (len(scores) < EPISODES_train ):
            time.sleep(200) # main thread saves the model every 200 sec
            if (len(scores)>10):
                self.save_model('pypow_14_a3c')
                print("saved NN model at episode", episode, "\n")

    def save_model(self, name):
        self.actor.save_weights(name + "_actor.h5")
        self.critic.save_weights(name + "_critic.h5")

    def load_model(self, name):
        self.actor.load_weights(name + "_actor.h5")
        self.critic.load_weights(name + "_critic.h5")

# This is Agent(local) class for threading
class Agent(threading.Thread):
    def __init__(self, index, actor, critic, optimizer, env_name, discount_factor, action_size, state_size):
        threading.Thread.__init__(self)

        self.states = []
        self.rewards = []
        self.actions = []

        self.index = index
        self.actor = actor
        self.critic = critic
        self.optimizer = optimizer
        self.env_name = env_name
        self.discount_factor = discount_factor
        self.action_size = action_size
        self.state_size = state_size

    # Thread interactive with environment
    def run(self):
        global episode
        global episode_test
        env = set_environement(start_id=self.index)
        while episode < EPISODES_train:
            state = env.reset()
            state_obs = observation_space.array_to_observation(state)
            state = self.useful_state(state)
            time_hour = 0
            score = 0
            time_step = 0
            non_zero_actions = 0
            while True:
                if min(state_obs.ampere_flows < 0.6*state_obs.thermal_limits):
                    action = 0
                else:
                    action = self.get_action(env,state)
                next_state, reward, done, flag = env.step(valid_actions_array_uniq[action,:])
                if done:
                    score += -1000 # this is the penalty for grid failure.
                    self.memory(state, action, -1000)
                else:
                    state_obs = observation_space.array_to_observation(next_state)
                    time_hour = state_obs.date_day*10000 + state_obs.date_hour * 100+ state_obs.date_minute
                    current_lim_factor = 0.85
                    over_current = 50 * sum(((state_obs.ampere_flows - current_lim_factor * state_obs.thermal_limits ) / (state_obs.thermal_limits))[
                        state_obs.ampere_flows > current_lim_factor * state_obs.thermal_limits]) # # penalizing lines close to the limit
                    score += (reward-over_current)
                    self.memory(state, action, (reward - over_current))
                non_zero_actions += 0 if action==0 else 1
                state = self.useful_state(next_state) if not done else np.zeros([1, state_size])
                time_step += 1
                if time_step % training_batch_size ==0:
                    print("Continue Thread:", self.index, "/ train episode: ", episode, "/ score : ", int(score),
                          "/ with recent time:", time_step, "/ with recent action", action,"/ number of non-zero actions", non_zero_actions, "/ day_hour_min:", time_hour)
                    self.train_episode(score < 2000000) # max score = 80000
                if done or time_step > time_step_end:
                    if done:
                        print("----STOPPED Thread:", self.index, "/ train episode: ", episode, "/ score : ", int(score),
                              "/ with final time:", time_step, "/ with final action", action,
                              "/ number of non-zero actions", non_zero_actions, "/ day_hour_min:", time_hour)
                    if time_step > time_step_end:
                        print("End Thread:", self.index, "/ train episode: ", episode, "/ score : ", int(score),
                              "/ with final time:", time_step, "/ with final action", action,
                              "/ number of non-zero actions", non_zero_actions, "/ day_hour_min:", time_hour)
                    scores.append(score)
                    episode += 1
                    self.train_episode(score < 2000000) # max score = 80000
                    break

    # In Policy Gradient, Q function is not available.
    # Instead agent uses sample returns for evaluating policy
    def discount_rewards(self, rewards, done=True):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        if not done:
            running_add = self.critic.predict(np.reshape(self.states[-1], (1, self.state_size)))[0]
        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.discount_factor + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

    # save <s, a ,r> of each step
    # this is used for calculating discounted rewards
    def memory(self, state, action, reward):
        self.states.append(state)
        act = np.zeros(self.action_size)
        act[action] = 1
        self.actions.append(act)
        self.rewards.append(reward)

    # update policy network and value network every episode
    def train_episode(self, done):
        discounted_rewards = self.discount_rewards(self.rewards, done)

        values = self.critic.predict(np.array(self.states))
        values = np.reshape(values, len(values))

        advantages = discounted_rewards - values

        self.optimizer[0]([self.states, self.actions, advantages])
        self.optimizer[1]([self.states, discounted_rewards])
        self.states, self.actions, self.rewards = [], [], []

    def useful_state(self,full_obs_state): # extracts the useful states from the full observation
        obs_structure = observation_space.array_to_observation(full_obs_state)
        selected_obs = np.hstack((obs_structure.lines_or_nodes.astype(int),obs_structure.lines_ex_nodes.astype(int)))
        selected_obs = np.hstack((selected_obs,obs_structure.loads_nodes.astype(int)))
        selected_obs = np.hstack((selected_obs,obs_structure.productions_nodes.astype(int)))
        selected_obs = np.hstack((selected_obs,obs_structure.lines_status.astype(int)))
        selected_obs = np.hstack((selected_obs,obs_structure.active_productions/100))
        selected_obs = np.hstack((selected_obs,obs_structure.active_loads/100))
        selected_obs = np.hstack((selected_obs,obs_structure.reactive_loads/100))
        selected_obs = np.hstack((selected_obs,obs_structure.voltage_productions))
        selected_obs = np.hstack((selected_obs,obs_structure.thermal_limits/100))
        selected_obs = np.hstack((selected_obs,obs_structure.ampere_flows/100))
        selected_obs = np.hstack((selected_obs,obs_structure.date_hour.astype(int))) # /24
        selected_obs = np.hstack((selected_obs,obs_structure.date_minute.astype(int))) # /60
        selected_obs = np.hstack((selected_obs,obs_structure.timesteps_before_nodes_reactionable.astype(int)))
        return selected_obs

    def get_action(self, env, state):
        policy_nn = self.actor.predict(np.reshape(state, [1, self.state_size]))[0]
        policy_nn_subid_mask = policy_nn * (1 - valid_actions_masking_subid_perm.dot((state[-14:]>0).astype(int))) # this masking prevents any illegal operation
        policy_chosen_list = np.random.choice(self.action_size, 4, replace=True,
                                              p=policy_nn_subid_mask / sum(policy_nn_subid_mask))
        policy_chosen_list = np.hstack((0, policy_chosen_list)) # adding no action option # comment this line as agent learns...
        obs_0, rw_0, done_0, _  = env.simulate(valid_actions_array_uniq[policy_chosen_list[0],:])
        obs_1, rw_1, done_1, _  = env.simulate(valid_actions_array_uniq[policy_chosen_list[1],:])
        obs_2, rw_2, done_2, _  = env.simulate(valid_actions_array_uniq[policy_chosen_list[2],:])
        obs_3, rw_3, done_3, _  = env.simulate(valid_actions_array_uniq[policy_chosen_list[3],:])
        rw_0 = self.est_reward_update(obs_0, rw_0, done_0)
        rw_1 = self.est_reward_update(obs_1, rw_1, done_1)
        rw_2 = self.est_reward_update(obs_2, rw_2, done_2)
        rw_3 = self.est_reward_update(obs_3, rw_3, done_3)
        return policy_chosen_list[np.argmax([rw_0,rw_1,rw_2,rw_3])]


    def est_reward_update(self,obs,rw,done): # penalizing overloaded lines
        obs = observation_space.array_to_observation(obs) if not done else 0
        rw_0 = rw - 5000 * sum(((0.95 * obs.thermal_limits - obs.ampere_flows) / (obs.thermal_limits))[
                            obs.ampere_flows > 0.95 * obs.thermal_limits]) if not done else -100
        return rw_0

if __name__ == "__main__":
    loaded = np.load('valid_actions_array_uniq.npz')
    valid_actions_array_uniq = loaded['valid_actions_array_uniq']  # this has 157 actions
    loaded_sub_d_mask = np.load('valid_actions_masking_subid_perm.npz')
    valid_actions_masking_subid_perm = loaded_sub_d_mask['valid_actions_masking_subid_perm'] # this maps the substation IDs with the actions

    env = set_environement()
    state_size = 164
    action_size = 157
    observation_space = env.observation_space
    del env
    env_name = 'pypow_14'
    global_agent = A3CAgent(state_size, action_size,env_name)
    global_agent.train()
