import numpy as np
import pickle

if __name__ == "__main__":
    # TODO uncomment this when submitting
    # train_x = np.loadtxt("train_x")
    # train_y = np.loadtxt("train_y")
    # test_x = np.loadtxt("test_x")

    # TODO delete this when submitting
    pickle_in = open("fast_data.txt", "rb").read()
    data = pickle.loads(pickle_in)
    train_x = np.array(data[0])
    train_y = np.array(data[1])
    test_x = np.array(data[2])
