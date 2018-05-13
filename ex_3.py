import numpy as np
import pickle


def combine_examples(examples, labels):
    paired_examples = np.empty(labels.size, dtype=tuple)

    for i in xrange(labels.size):
        paired_examples[i] = (examples[i], labels[i])

    return paired_examples


def create_weights_matrix(input_size, hidden_size):
    pass


def create_bias_vector(hidden_size):
    pass


def generate_weights():
    input_layer_size = 10
    hidden_layer_size = 50
    output_layer_size = 784
    W1 = create_weights_matrix(hidden_layer_size, input_layer_size)
    b1 = create_bias_vector(hidden_layer_size)
    W2 = create_weights_matrix(output_layer_size, hidden_layer_size)
    b2 = create_bias_vector(hidden_layer_size)

    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}


def forward(x, activation_function, params):
    pass


def train(params, epochs, eta, activation_function, training_set, validation_set):

    for e in xrange(epochs):
        sum_loss = 0.0
        np.random.shuffle(training_set)

        for x, y in training_set:
            y_hat = forward(x, activation_function, params)


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

    # Combine examples and labels into tuples.
    paired_examples = combine_examples(train_x, train_y)

    # Shuffle examples and labels.
    np.random.shuffle(paired_examples)

    # Divide the pairs to training and validation sets 80:20.
    validation_size = int(paired_examples.size * 0.2)
    validation_set = paired_examples[-validation_size:]
    training_set = paired_examples[:-validation_size]

    # Generate initial weights.
    params = generate_weights()

    # Initialize hyper-parameters.
    epochs = 30
    eta = 0.01
    sigmoid = lambda x: 1 / (1 + np.exp(-x))
