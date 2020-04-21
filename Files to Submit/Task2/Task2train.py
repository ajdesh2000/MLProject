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



class trainTask2():

	
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


	def neural_network_model(self,input_size, output_size):

		model = Sequential()
		
		model.add(Dense(24, input_dim=input_size,activation='relu'))
		model.add(Dense(24, activation='relu'))
		model.add(Dense(output_size,activation='linear'))
		
		model.compile(loss='mse', optimizer=Adam(lr=0.001), metrics=['accuracy'])
		
		return model


	def replay(self, model, training_data, TRAIN_SIZE):
		global EPSILON, EPSILON_MIN, EPSILON_DECAY
		
		batch_data = random.sample(training_data, TRAIN_SIZE)
		for t in batch_data:
			state, reward, done, new_state, action = t
			
			yi = model.predict(state)
			
			if(done):
				yi[0][action] = reward
			else:
				a = model.predict(new_state)[0]
				t = self.target_model.predict(new_state)[0]
				yi[0][action] = reward + self.GAMMA * t[np.argmax(a)]
			model.fit(state, yi, epochs=1, verbose=0)
		if self.EPSILON > self.EPSILON_MIN:
			self.EPSILON *= self.EPSILON_DECAY

		return model



	def next_action(self, state, output_size):
		if(np.random.rand() <= self.EPSILON): #Explore: Random action
			return random.randrange(output_size)
		return np.argmax(self.model.predict(state)) #Exploitation: Action with best possible reward

	
	def train(self):
		self.LR = 1e-3
		self.input_size = self.env.observation_space.shape[0]
		self.output_size = self.env.action_space.n
		
		self.model = self.neural_network_model(self.input_size, self.output_size)
		self.target_model = self.neural_network_model(self.input_size, self.output_size)
		self.model.summary()


		scores = []
		game_memory = deque(maxlen=100000)

		N = 100
		last_N_scores = deque(maxlen=N)
		SCORE_REQUIRED = 490

		GOAL_STEPS = 500
		TRAIN_EPISODES = 100000

		TRAIN_SIZE = 32
		UPDATE_EVERY = 1

		self.EPSILON = 1.0 #Defines the exploration range
		self.EPSILON_MIN = 0.01
		self.EPSILON_DECAY = 0.99
		self.GAMMA = 0.99

		for game in range(TRAIN_EPISODES):
			state = self.env.reset()
			
			if game % UPDATE_EVERY == 0:
				self.target_model.set_weights(self.model.get_weights())
			
			score = 0
			for _ in range(GOAL_STEPS):
				#env.render()
				state = state.reshape(-1, len(state))
				
				action = self.next_action(state, self.output_size)
						
				new_state, reward, done, info = self.env.step(action)
				
				score += reward
				if done:
					reward = -reward
				else:
					reward = reward
				
				game_memory.append((state, reward, done, new_state.reshape(-1, len(new_state)), action))
				
				state = new_state
				if done: 
					#print("episode: {}/{}, score: {}, e: {:.2}".format(game, EPISODES, _, EPSILON))
					break
					
			#complete training if last N episodes yield more than required score
			last_N_scores.append(score)
			if(sum(last_N_scores)/N >= SCORE_REQUIRED):
				print('Score {} achieved at {} game'.format(SCORE_REQUIRED, game))
				break
					
			if(len(game_memory) > TRAIN_SIZE):
				self.replay(self.model, game_memory, TRAIN_SIZE)
			
			if(game % 100 == 0):
				print('Game {} score : {}'.format(game, score))
				scores.append(score)
				
			
		if(len(scores) > 0):
			print('Average Score taken every 100 steps:',sum(scores)/len(scores))
		else:
			print('Average Score:', SCORE_REQUIRED)



		self.plot_res(scores)

		self.model.save('model2.h5')

if __name__ == "__main__":
	trainer=trainTask2()
	trainer.train()



