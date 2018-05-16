import numpy as np
import pickle


def main():
    # TODO uncomment this when submitting
    # train_x = np.loadtxt("train_x")
    # train_y = np.loadtxt("train_y")
    # test_x = np.loadtxt("test_x")

    # TODO delete this when submitting
    pickle_in = open("fast_data.txt", "rb").read()
    data = pickle.loads(pickle_in)
    train_x = np.array(data[0]) / 255.0
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

    train_network(params, epochs, eta, sigmoid, sigmoid_derivative, training_set, validation_set)


def combine_examples(examples, labels):
    """
    Combines the examples and labels into one set.
    :param examples: examples list
    :param labels: labels list
    :return: combined list
    """
    paired_examples = np.empty(labels.size, dtype=tuple)

    for i in xrange(labels.size):
        paired_examples[i] = (examples[i], labels[i])

    return paired_examples


def create_weights_matrix(rows, columns=1):
    """
    Creates a weights matrix with uniformly distributed values.
    :param rows: number of rows
    :param columns: number of columns
    :return: weights matrix
    """
    # TODO change the low and high values, maybe use random instead of uniform
    weights_matrix = np.empty(shape=(rows, columns))

    for row in xrange(rows):
        weights_matrix[row, :] = np.random.uniform(-0.08, 0.08, columns)

    return weights_matrix


def generate_weights():
    """
    Creates the needed weights for the network.
    :return: dictionary of weights
    """
    input_layer_size = 784
    hidden_layer_size = 50
    output_layer_size = 10
    W1 = create_weights_matrix(hidden_layer_size, input_layer_size)
    b1 = create_weights_matrix(hidden_layer_size)
    W2 = create_weights_matrix(output_layer_size, hidden_layer_size)
    b2 = create_weights_matrix(hidden_layer_size)

    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}


def softmax(x):
    """
    Softmax function.
    :param x: input
    :return: output
    """
    z = np.exp(x - np.amax(x))
    return z / np.sum(z)


def sigmoid(x):
    """
    Sigmoid function.
    :param x: input
    :return: output
    """
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_derivative(x):
    """
    Sigmoid derivative.
    :param x: input
    :return: output
    """
    return x * (1.0 - x)


def forward_propagation(x, activation_function, params):
    """
    Performs forward propagation.
    :param x: example
    :param activation_function: activation function
    :param params: parameters
    :return: output
    """
    W1, b1, W2, b2 = [params[key] for key in ('W1', 'b1', 'W2', 'b2')]
    z1 = np.dot(W1, x) + b1
    h1 = activation_function(z1)
    z2 = np.dot(W2, h1) + b2
    y_hat = softmax(z2)
    ret = {'x': x, 'z1': z1, 'h1': h1, 'z2': z2, 'y_hat': y_hat}

    # Add the values of params to the returned value.
    for key in params:
        ret[key] = params[key]
    return ret


def one_hot_vector(y, num_of_elements):
    """
    Creates a one hot vector.
    :param y: label
    :param num_of_elements: number of elements in the vector
    :return: one hot vector
    """
    vector = np.zeros(num_of_elements, dtype=int)
    vector[y] = 1
    return vector


def calculate_loss(y, y_hat):
    # TODO maybe no need to use one hot vector, just do y*log(y_hat) instead of the sum
    """
    Calculates the loss.
    :param y: true label
    :param y_hat: predicted label
    :return: loss value
    """
    y = one_hot_vector(y, 10)
    loss = 0

    for i in xrange(y.size):
        loss += y[i] * np.log(y_hat[i])
    return -loss


def back_propagation(x, y, forward_cache, activation_function_derivative):
    """
    Performs the back propagation algorithm for one hidden layer.
    :param x: example
    :param y: true label
    :param forward_cache: parameters dictionary
    :param activation_function_derivative: activation function derivative
    :return: gradients dictionary
    """
    W1, W2, h1, y_hat = [forward_cache[key] for key in ('W1', 'W2', 'h1', 'h_hat')]

    # Calculation of the dW2 and db2 gradients.
    # TODO check if global is needed here
    dz2 = y_hat
    dz2[y] -= 1
    dW2 = np.outer(dz2, h1)
    db2 = dz2

    # Calculation of the dW1 and db1 gradients
    dz1 = np.dot(W2, dz2) * activation_function_derivative(h1)
    dW1 = np.outer(dz1, x)
    db1 = dz1

    return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}


def update(eta, forward_cache, gradients):
    """
    Updates the weights using the sgd update rule.
    :param eta:
    :param forward_cache:
    :param gradients:
    :return:
    """
    W2, b2, W1, b1 = [forward_cache[key] for key in ("W2", "b2", "W1", "b1")]
    dW2, db2, dW1, db1 = [gradients[key] for key in ("dW2", "db2", "dW1", "db1")]

    W2 = W2 - eta * dW2
    b2 = b2 - eta * db2
    W1 = W1 - eta * dW1
    b1 = b1 - eta * db1

    return {"W2": W2, "b2": b2, "W1": W1, "b1": b1}


def predict_on_validation(validation_set, params, activation_function):
    # TODO change variables names, and in general change the code structure
    good = 0.0
    sum_loss = 0.0

    for x, y in validation_set:
        forward_cache = forward_propagation(x, activation_function, params)
        loss = calculate_loss(y, forward_cache["y_hat"])
        sum_loss += loss

        if np.argmax(forward_cache["y_hat"]) == y:
            good += 1

    accuracy = good / validation_set[0].shape[0]
    average_loss = sum_loss / validation_set[0].shape[0]

    return average_loss, accuracy


def train_network(params, epochs, eta, activation_function, activation_function_derivative, training_set,
                  validation_set):
    for e in xrange(epochs):
        sum_loss = 0.0
        np.random.shuffle(training_set)

        for x, y in training_set:
            forward_cache = forward_propagation(x, activation_function, params)
            loss = calculate_loss(y, forward_cache["y_hat"])
            sum_loss += loss
            gradients = back_propagation(x, y, forward_cache, activation_function_derivative)
            params = update(eta, forward_cache, gradients)

        validation_loss, accuracy = predict_on_validation(validation_set, params, activation_function)

        print e, sum_loss / training_set[0].shape[0], validation_loss, "{}%".format(accuracy * 100)


if __name__ == "__main__":
    main()
