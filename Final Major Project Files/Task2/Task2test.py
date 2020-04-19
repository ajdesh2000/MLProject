import sys
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






class testTasks():
	
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



	def next_action(self, state, output_size):
		if(np.random.rand() <= self.epsilon): #Explore: Random action
			act = [0] * output_size
			act[np.random.randint(0, output_size)] = 1
			act = np.array(act, dtype=np.float32)
			return act
		return self.a_model.predict(np.expand_dims(state, axis=0))[0] #Exploitation: Action with best possible reward


	#Initializiing models with target models(For slow update in DDQL)
	def test(self):
		self.LEARNING_RATE = 0.001

			
		_, self.a_model = self.actor_model(self.env.observation_space.shape, self.env.action_space.n)
		self.a_model.load_weights('task2_a_model.h5')


		if(len(sys.argv)>1):
			TEST_EPISODES=int(sys.argv[1])
		else:
			TEST_EPISODES = 100	
		print('Taking ',TEST_EPISODES,' test episodes.')

		goal_steps = 500
		self.epsilon = 0.0
		output_size = self.env.action_space.n

		scores = []
		for episode in range(TEST_EPISODES):
			state = self.env.reset()
			
			score = 0
			for _ in range(goal_steps):
				
				action = self.next_action(state, output_size)
				
				new_state, reward, done, info = self.env.step(np.argmax(action))
				score += reward
				
				state = new_state
				if(done):
					break
					
			scores.append(score)
				
		self.plot_res(scores)        
		print("Average score over {} episodes: {}".format(TEST_EPISODES, sum(scores)/len(scores)))
		
if __name__ == "__main__":
	tester=testTasks()
	tester.test()

