import sys
import gym
import random
import numpy as np
from statistics import median, mean
from collections import Counter
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, LSTM
from keras.optimizers import Adam
from keras import backend as K
import matplotlib.pyplot as plt
from IPython.display import clear_output
from collections import deque


class testTask2():

	
	def __init__(self):
		self.env = gym.make("CartPole-v1")

	


	def plot_res(self, values, title=''):   
		clear_output(wait=True)
		f, ax = plt.subplots(nrows=1, ncols=2, figsize=(12,5))
		f.suptitle(title)
		ax[0].plot(values, label='Score per run')
		ax[0].axhline(100, c='red',ls='--', label='goal')
		ax[0].set_xlabel('Episodes')
		ax[0].set_ylabel('Reward')
		x = range(len(values))
		ax[0].legend()
		try:
			z = np.polyfit(x, values, 1)
			p = np.poly1d(z)
			ax[0].plot(x,p(x),"--", label='trend')
		except:
			print('')
		ax[1].hist(values[-50:])
		ax[1].axvline(500, c='red', label='goal')
		ax[1].set_xlabel('Scores per Last 50 Episodes')
		ax[1].set_ylabel('Frequency')
		ax[1].legend()
		plt.show()


	def neural_network_model(self,input_size, OUTPUT_SIZE):

		model = Sequential()
		
		model.add(Dense(24, input_dim=input_size,activation='relu'))
		model.add(Dense(24, activation='relu'))
		model.add(Dense(OUTPUT_SIZE,activation='linear'))
		
		model.compile(loss='mse', optimizer=Adam(lr=0.001), metrics=['accuracy'])
		
		return model


	def next_action(self, state, OUTPUT_SIZE):
		if(np.random.rand() <= self.EPSILON): #Explore: Random action
			return random.randrange(OUTPUT_SIZE)
		return np.argmax(self.model.predict(state)) #Exploitation: Action with best possible reward

	
	def test(self):
		self.LR = 1e-3
		self.input_size = self.env.observation_space.shape[0]
		self.OUTPUT_SIZE = self.env.action_space.n
		
		
		self.model = load_model('model2.h5')

		if(len(sys.argv)>1):
			TEST_EPISODES=int(sys.argv[1])
		else:
			TEST_EPISODES = 100	
		print('Taking ',TEST_EPISODES,' test episodes.')
		
		GOAL_STEPS = 500
		self.EPSILON = 0.0
		OUTPUT_SIZE = 2

		scores = []
		for episode in range(TEST_EPISODES):
			state = self.env.reset()
			
			score = 0
			for _ in range(GOAL_STEPS):
				#self.env.render()
				
				state = state.reshape(-1, len(state))
				action = self.next_action(state, self.OUTPUT_SIZE)
				
				new_state, reward, done, info = self.env.step(action)
				score += reward
				
				state = new_state
				if(done):
					break
					
			scores.append(score)
				
		self.plot_res(scores)        
		print("Average score over {} episodes: {}".format(TEST_EPISODES, sum(scores)/len(scores)))


if __name__ == "__main__":
	tester=testTask2()
	tester.test()



