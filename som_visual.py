import numpy as np
from minisom import MiniSom

data = np.load("reliability_vectors.npy")

som = MiniSom(10, 10, data.shape[1], sigma=1.0, learning_rate=0.5)
som.random_weights_init(data)
som.train(data, 1000)

import pickle
pickle.dump(som, open("som_model.pkl", "wb"))

print("SOM trained.")