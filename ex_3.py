import numpy as np
import pickle


def main():
    # TODO uncomment this when submitting
    # train_x = np.loadtxt("train_x") / 255.0
    # train_y = np.loadtxt("train_y")
    # test_x = np.loadtxt("test_x") / 255.0

    # TODO delete this when submitting
    pickle_in = open("fast_data.txt", "rb").read()
    data = pickle.loads(pickle_in)
    train_x = np.array(data[0]) / 255.0
    train_y = np.array(data[1])
    # TODO maybe divide by 255. as well
    test_x = np.array(data[2]) / 255.0

    # Combine examples and labels into tuples.
    paired_examples = combine_examples(train_x, train_y)

    # Shuffle examples and labels.
    np.random.shuffle(paired_examples)

    # Divide the pairs to training and validation sets 80:20.
    validation_size = int(paired_examples.size * 0.2)
    validation_set = paired_examples[-validation_size:]
    training_set = paired_examples[:-validation_size]

    # Initialize hyper-parameters.
    epochs = 30
    eta = 0.005
    hidden_layer_size = 100

    # Generate initial weights.
    params = generate_weights(hidden_layer_size)

    # Train the network.
    trained_params = train_network(params, epochs, eta, sigmoid, sigmoid_derivative,
                                   training_set, validation_set)

    # Perform a prediction over the test set, and write it to a file.
    write_prediction(trained_params, test_x)


def write_prediction(trained_params, test_x):
    """
    Performs a prediction over the test set and writes it to a file.
    :param trained_params: trained parameters
    :param test_x: test set
    :return: None
    """
    with open("test.pred", "w") as test_file:
        for x in test_x[:-1]:

            # Pass the example through the classifier.
            output = forward_propagation(x, sigmoid, trained_params)

            # Extract the predicted label.
            predicted_value = str(np.argmax(output["y_hat"]))
            test_file.write(predicted_value + '\n')

        # Avoid writing blank line at the end of file.
        output = forward_propagation(test_x[-1], sigmoid, trained_params)
        predicted_value = str(np.argmax(output["y_hat"]))
        test_file.write(predicted_value)


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


def generate_weights(hidden_layer_size):
    """
    Creates the needed weights for the network.
    :return: dictionary of weights
    """
    input_layer_size = 784
    output_layer_size = 10
    W1 = np.random.uniform(-0.08, 0.08, (hidden_layer_size, input_layer_size))
    b1 = np.random.uniform(-0.08, 0.08, (hidden_layer_size, 1))
    W2 = np.random.uniform(-0.08, 0.08, (output_layer_size, hidden_layer_size))
    b2 = np.random.uniform(-0.08, 0.08, (output_layer_size, 1))

    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}


def softmax(x):
    """
    Softmax function.
    :param x: input
    :return: output
    """
    z = np.exp(x - np.max(x))
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


def tanh(x):
    """
    Hyperbolic tangens function.
    :param x: input
    :return: output
    """
    return np.tanh(x)


def tanh_derivative(x):
    """
    Hyperbolic tangens derivative.
    :param x: input
    :return: output
    """
    return 1 - (x ** 2)


def relu(x):
    """
    ReLU function.
    :param x: input
    :return: output
    """
    return x * (x > 0)


def relu_derivative(x):
    """
    ReLU derivative.
    :param x: input
    :return: output
    """
    return 1.0 * (x > 0)


def forward_propagation(x, activation_function, params):
    """
    Performs forward propagation.
    :param x: example
    :param activation_function: activation function
    :param params: parameters
    :return: output
    """
    W1, b1, W2, b2 = [params[key] for key in ('W1', 'b1', 'W2', 'b2')]
    x = np.array(x, ndmin=2).T
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
    vector[int(y)] = 1
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
    loss = 0.0

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
    W1, W2, h1, y_hat = [forward_cache[key] for key in ('W1', 'W2', 'h1', 'y_hat')]

    # Calculation of the dW2 and db2 gradients.
    # TODO check if global is needed here
    dz2 = y_hat
    dz2[int(y)] -= 1
    dW2 = np.outer(dz2, h1.T)
    db2 = dz2

    # Calculation of the dW1 and db1 gradients
    dz1 = np.dot(W2.T, dz2) * activation_function_derivative(h1)
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
    """
    Performs a prediction over the validation set.
    :param validation_set: validation set
    :param params: parameters dictionary
    :param activation_function: activation function
    :return: average loss, accuracy
    """
    # TODO change variables names, and in general change the code structure
    # Counts the number of times the classifier was correct.
    correct = 0.0

    # Sums the loss.
    sum_loss = 0.0
    num_of_examples = validation_set.shape[0]

    # Go over the validation set, and check how accurate the classifier is.
    for x, y in validation_set:
        forward_cache = forward_propagation(x, activation_function, params)
        loss = calculate_loss(y, forward_cache["y_hat"])
        sum_loss += loss

        # Check if the classifier's output matches the correct label.
        if np.argmax(forward_cache["y_hat"]) == y:
            correct += 1

    # Calculate the accuracy of the classifier.
    accuracy = correct / num_of_examples

    # Calculate the average loss.
    average_loss = sum_loss / num_of_examples

    return average_loss, accuracy


def train_network(params, epochs, eta, activation_function, activation_function_derivative, training_set,
                  validation_set):
    """
    Trains the neural network.
    :param params: weights matrices and vectors
    :param epochs: number of epochs
    :param eta: learning rate
    :param activation_function: activation function
    :param activation_function_derivative: activation function derivative
    :param training_set: training set
    :param validation_set: validation set
    :return: classifier
    """
    for e in xrange(epochs):
        sum_loss = 0.0
        np.random.shuffle(training_set)

        for x, y in training_set:

            # Perform forward propagation.
            forward_cache = forward_propagation(x, activation_function, params)

            # Calculate the loss.
            loss = calculate_loss(y, forward_cache["y_hat"])
            sum_loss += loss

            # Perform back propagation.
            gradients = back_propagation(x, y, forward_cache, activation_function_derivative)

            # Update the weights.
            params = update(eta, forward_cache, gradients)

        # Check the classifier over the validation set.
        validation_loss, accuracy = predict_on_validation(validation_set, params, activation_function)

        print "epoch: {0} train loss: {1} dev loss: {2} accuracy: {3}%".format(e, sum_loss / training_set.shape[0],
                                                                               validation_loss, accuracy * 100)

    return params


if __name__ == "__main__":
    main()
