from __future__ import division
import argparse

from PIL import Image
import numpy as np
import gym
import random

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Convolution2D, Permute, Conv2D
from keras.optimizers import Adam
import keras.backend as K

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint


# Load the enviroment
env = gym.make('LunarLander-v2')

sd = 555
np.random.seed(sd)
random.seed(sd)
env.seed(sd)

nb_actions = env.action_space.n

# Define the model
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape)) # Input layer
model.add(Dense(512)) # Layer 1
model.add(Activation('relu'))
model.add(Dense(256)) # Layer 2
model.add(Activation('relu'))
model.add(Dense(128)) # Layer 3
model.add(Activation('relu'))
model.add(Dense(nb_actions)) # Output Layer
model.add(Activation('linear'))
print(model.summary())

memory = SequentialMemory(limit=300000, window_length=1)
# Define the policy
policy = EpsGreedyQPolicy()

# Define the DQN Agent and load the model in it
dqn = DQNAgent(model=model, nb_actions=nb_actions, policy=policy, memory=memory, nb_steps_warmup=50000, target_model_update=1e-2)

# Compile the model
dqn.compile(Adam(lr=1e-5), metrics=['mae'])

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['train', 'test'], default='train')
parser.add_argument('--env-name', type=str, default='LunarLander-v2')
parser.add_argument('--weights', type=str, default=None)
args = parser.parse_args()

if args.mode == 'train':
    # For fitting the model in training phase
    ''' 
        Type below command to run
        python dqn_lunarlander.py
    '''
    weights_filename = 'dqn_{}_weights.h5f'.format(args.env_name)
    checkpoint_weights_filename = 'dqn_' + args.env_name + '_weights_{step}.h5f' #Trained Weights file
    log_filename = 'dqn_{}_log.json'.format(args.env_name) # JSON log file
    callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=250000)]
    callbacks += [FileLogger(log_filename, interval=100)]
    # Fit the model and add callbacks for log generation
    dqn.fit(env, nb_steps=1550000, log_interval=10000, visualize=False, callbacks=callbacks)

    # After training is done, we save the final weights.
    dqn.save_weights(weights_filename, overwrite=True)

elif args.mode == 'test':
    # For testing the model in experiment test phase
    '''
       Type below command to run
       python dqn_lunarlander.py --mode test --weights dqn_LunarLander-v2_weights.h5f
    '''
    if args.weights:
        weights_filename = args.weights
    # Load the weights from passed filename
    dqn.load_weights(weights_filename)
    # Test for 200 episodes
    dqn.test(env, nb_episodes=200, visualize=True)
