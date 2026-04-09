import numpy as np
from minisom import MiniSom
import pickle

print("Loading reliability vectors...")

data = np.load("reliability_vectors.npy")

print("Data shape:", data.shape)

print("Training SOM...")

som = MiniSom(
    x=10,
    y=10,
    input_len=data.shape[1],
    sigma=1.0,
    learning_rate=0.5
)

som.random_weights_init(data)
som.train_random(data, 2000)

pickle.dump(som, open("models/som_model.pkl", "wb"))

print("SOM training complete.")