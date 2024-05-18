import numpy as np
import gymnasium as gym

# add hidden layers or nodes according to needs
IL = 4 #input layer nodes
HL = 20 #hidden layer nodes
OL = 2 #output layer nodes
w1 = np.random.randn(HL,IL) / np.sqrt(IL)
w2 = np.random.randn(OL,HL) / np.sqrt(HL)
NumWeights1 = len(w1.flatten())
NumWeights2 = len(w2.flatten())

#forward propagation
def predict(s,w1,w2):
    h = np.dot(w1,s) #input to hidden layer
    h[h<0]=0 #relu
    out = np.dot(w2,h) #hidden layer to output
    out = 1.0 / (1.0 + np.exp(-out)) #sigmoid 
    return out

#load environment
# This was v0, but it was giving me warnings its deprecated so i upgraded it
env = gym.make('CartPole-v1')
#parameters
NumEpisodes = 50
NumPolicies = 10
sigma = 0.1
learning_rate = 0.001

Reward = np.zeros(NumPolicies)


#start learning
for episode in range(NumEpisodes):
    #generate random variations around original policy
    eps = np.random.randn(NumPolicies, NumWeights1+NumWeights2)
    #evaluate each policy over one episode
    for policy in range(NumPolicies):
        w1_try = w1 + sigma * eps[policy, :NumWeights1].reshape(w1.shape)
        w2_try = w2 + sigma * eps[policy, NumWeights1:].reshape(w2.shape)

        #initial state
        observation = env.reset()[0]
        #observe initial state
        Reward[policy] = 0
        while True:
            Action = predict(observation, w1_try, w2_try)
            Action = np.argmax(Action)
            #execute action
            observation_new, reward, done, _, _ = env.step(Action)
            #collect reward
            Reward[policy] += reward
            #update state
            observation = observation_new
            #end episode
            if done:
                break

#calculate incremental rewards
F = (Reward - np.mean(Reward))
print("Reward:", Reward)
print("F:", F)


#update weights of original policy according to rewards of all variations
weights_update = learning_rate/(NumPolicies*sigma) * np.dot(eps.T, F)
w1 += weights_update[:NumWeights1].reshape(w1.shape)
w2 += weights_update[NumWeights1:].reshape(w2.shape)






