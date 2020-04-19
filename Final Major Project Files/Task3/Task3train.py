import gym
import random
import numpy as np
from statistics import median, mean
from collections import Counter
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Dropout, LSTM, Input
from keras.optimizers import Adam
from keras.layers.merge import Add

import tensorflow.compat.v1 as tf
from tensorflow.compat.v1.keras import backend as K
tf.disable_eager_execution()

import matplotlib.pyplot as plt
from IPython.display import clear_output
from collections import deque




class trainTask2():

	
	
	def __init__(self):
		self.env = gym.make("CartPole-v1")


	#Plot score over episodes
	def plot_res(self,values, title=''):   
		clear_output(wait=True)		
		# Define the figure
		f, ax = plt.subplots(nrows=1, ncols=2, figsize=(12,5))
		f.suptitle(title)
		ax[0].plot(values, label='score per run')
		ax[0].axhline(500, c='red',ls='--', label='goal')
		ax[0].set_xlabel('Episodes')
		ax[0].set_ylabel('Reward')
		x = range(len(values))
		ax[0].legend()
		# Calculate the trend
		try:
			z = np.polyfit(x, values, 1)
			p = np.poly1d(z)
			ax[0].plot(x,p(x),"--", label='trend')
		except:
			print('')
		
		# Plot the histogram of results
		ax[1].hist(values[-50:])
		ax[1].axvline(500, c='red', label='goal')
		ax[1].set_xlabel('Scores per Last 50 Episodes')
		ax[1].set_ylabel('Frequency')
		ax[1].legend()
		plt.show()


	def actor_model(self,input_size, output_size):
		input_state = Input(shape=input_size)
		output=Dense(24, activation='relu')(input_state)
		output=Dense(24, activation='relu')(output)
		output=Dense(output_size, activation='softmax')(output)
		model = Model(input=input_state, output=output)
		model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=self.LEARNING_RATE), metrics=['accuracy'])
		print(model.summary())
		return input_state, model

	def critic_model(self,input_size_state, input_size_action):
		input_state = Input(shape=input_size_state)
		state_hidden = Dense(24, activation='relu')(input_state)
		state_hidden = Dense(24)(state_hidden)
		input_action = Input(shape=(input_size_action,))
		action_hidden = Dense(24)(input_action)
		action_hidden = Dense(24)(action_hidden)
		output = Add()([state_hidden, action_hidden])
		output = Dense(24, activation='relu')(output)
		output = Dense(1, activation='linear')(output)
		model = Model(input=[input_state,input_action], output=output)
		model.compile(loss='mse', optimizer=Adam(lr=self.LEARNING_RATE), metrics=['accuracy'])
		print(model.summary())
		return input_state, input_action, model




	def replay_actor(self,training_samples):		
		states = []
		actions = []
		for t in training_samples:
			state, reward, done, new_state, action = t
			act = self.a_model.predict(np.expand_dims(state, axis=0))[0]
			states.append(state)
			actions.append(act)
		states = np.array(states)
		actions = np.array(actions)			
		grads = self.sess.run(self.critic_grads, feed_dict={
			self.critic_input_state:  states, 
			self.critic_input_action: actions
		})[0]
		self.sess.run(self.optimize, feed_dict={
			self.actor_input_state: states,
			self.actor_critic_grads: grads
		})
        
		
		
		
	def replay_critic(self,training_samples):
		states=[]
		actions = []
		y=[]
		for t in training_samples:
			state, reward, done, new_state, action = t
			new_state = np.expand_dims(new_state, axis=0)
			if(done):
				reward=-reward
			else:
				if not self.double:
					#Deep q learning
					a = self.a_model.predict(new_state)
					t = self.c_model.predict([new_state, a])[0][0]
				else:
					#Double deep q learning
					a = self.a_target_model.predict(new_state)
					t = self.c_target_model.predict([new_state, a])[0][0]
				reward = reward + self.gamma * t
			states.append(state)
			actions.append(action)
			y.append(reward)
		states = np.array(states)
		actions = np.array(actions)
		y = np.expand_dims(np.array(y), axis=1)
		self.c_model.fit([states, actions], y, batch_size=len(training_samples), verbose=0)
    
	
	
	def replay(self,training_data, train_size):
		if(len(training_data) < train_size):
			return
		batch_data = random.sample(training_data, train_size)
		self.replay_critic(batch_data)
		self.replay_actor(batch_data)
		self.epsilon = max(self.epsilon*self.epsilon_decay, self.epsilon_min)

	def update_actor_target(self):
		self.a_target_model.set_weights(self.a_model.get_weights())
		
	def update_critic_target(self):
		self.c_target_model.set_weights(self.c_model.get_weights())

	def update_targets(self):
		self.update_actor_target()
		self.update_critic_target()

	def next_action(self, state, output_size):
		if(np.random.rand() <= self.epsilon): #Explore: Random action
			act = [0] * output_size
			act[np.random.randint(0, output_size)] = 1
			act = np.array(act, dtype=np.float32)
			return act
		return self.a_model.predict(np.expand_dims(state, axis=0))[0] #Exploitation: Action with best possible reward


	#Initializiing models with target models(For slow update in DDQL)
	def train(self):
		self.LEARNING_RATE = 0.001
	
		self.input_size = self.env.observation_space.shape
		self.output_size = self.env.action_space.n
		self.sess = tf.Session()
		K.set_session(self.sess)
		self.actor_critic_grads = tf.placeholder(tf.float32, [None, self.output_size])
		
		self.actor_input_state,self.a_model = self.actor_model(self.input_size, self.output_size)
		_,self.a_target_model = self.actor_model(self.input_size, self.output_size)
		self.critic_input_state, self.critic_input_action, self.c_model = self.critic_model(self.input_size, self.output_size)
		_, _, self.c_target_model = self.critic_model(self.input_size, self.output_size)

		self.actor_model_weights = self.a_model.trainable_weights
		self.actor_grads = tf.gradients(self.a_model.output, self.actor_model_weights, -self.actor_critic_grads)
		self.grads = zip(self.actor_grads, self.actor_model_weights)
		self.optimize = tf.train.AdamOptimizer(self.LEARNING_RATE).apply_gradients(self.grads)
		self.critic_grads = tf.gradients(self.c_model.output, self.critic_input_action)
		self.sess.run(tf.global_variables_initializer())


		scores = []
		game_memory = deque(maxlen=100000)
		SCORE_REQUIRED = 490
		N=100
		last_N_scores = deque(maxlen=N)

		SEED=42
		np.random.seed(SEED)
		random.seed(SEED)
		tf.set_random_seed(SEED)

		train_size = 32
		update_every = 1
		goal_steps = 500
		TRAIN_EPISODES = 500000

		self.epsilon = 1.0 #Defines the exploration range
		self.epsilon_min = 0.01
		self.epsilon_decay = 0.99
		self.gamma = 0.99

		self.double = False #Double deep q
		relearn = False

		if(relearn):
			epsilon = 0.01
			self.a_model.load_weights('task3_a_model.h5') #Loading weights for continued learning
			self.c_model.load_weights('task3_c_model.h5')

		for game in range(TRAIN_EPISODES):
			state = self.env.reset()
			
			#env.render()
			score = 0
			for _ in range(goal_steps):
				
				action = self.next_action(state, self.output_size)		
				new_state, reward, done, info = self.env.step(np.argmax(action))			
				score += reward
				game_memory.append((state, reward, done, new_state, action))
				state = new_state
				if done: 
					break
			
			if(len(game_memory) < 1000):
				continue
			
			self.replay(game_memory, train_size)
			
			if(self.double):
				if game % update_every == 0: #For double deep q learning
					update_targets()
			
			if(game % 100 == 0):
				print('Game {} score : {}, epsilon={:.2}'.format(game, score, self.epsilon))
				scores.append(score)
				
			#complete training if last N episodes yield more than required score
			last_N_scores.append(score)
			if(sum(last_N_scores)/N >= SCORE_REQUIRED):
				print('Score {} achieved at {} game'.format(SCORE_REQUIRED, game))
				break
				
			#Add randomness if the last N scores yield less than 11
			if(len(last_N_scores) == N and sum(last_N_scores)/(N) < 11 and self.epsilon == self.epsilon_min):
				self.epsilon = 1.0

		if(len(scores) > 0):
			print('Average Score taken every 100 steps:',sum(scores)/len(scores))
		else:
			print('Average Score:', SCORE_REQUIRED)



		self.a_model.save_weights('task3_a_model.h5')
		self.c_model.save_weights('task3_c_model.h5')

if __name__ == "__main__":
	trainer=trainTask2()
	trainer.train()

