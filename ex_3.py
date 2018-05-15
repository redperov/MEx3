import numpy as np
import pickle


def combine_examples(examples, labels):
    paired_examples = np.empty(labels.size, dtype=tuple)

    for i in xrange(labels.size):
        paired_examples[i] = (examples[i], labels[i])

    return paired_examples


def create_weights_matrix(rows, columns=1):
    # TODO change the low and high values
    weights_matrix = np.empty(shape=(rows, columns))

    for row in xrange(rows):
        weights_matrix[row, :] = np.random.uniform(-0.08, 0.08, columns)

    return weights_matrix


def generate_weights():
    input_layer_size = 784
    hidden_layer_size = 50
    output_layer_size = 10
    W1 = create_weights_matrix(hidden_layer_size, input_layer_size)
    b1 = create_weights_matrix(hidden_layer_size)
    W2 = create_weights_matrix(output_layer_size, hidden_layer_size)
    b2 = create_weights_matrix(hidden_layer_size)

    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}


def softmax(x):
    return (np.exp(x - np.amax(x))) / (np.sum(np.exp(x - np.amax(x))))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1.0 - x)


def forward_propagation(x, activation_function, params):
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
    vector = np.zeros(num_of_elements, dtype=int)
    vector[y] = 1
    return vector


def calculate_loss(y, y_hat):
    y = one_hot_vector(y, 10)
    loss = 0

    for i in xrange(y.size):
        loss += y[i] * np.log(y_hat[i])
    return -loss


def back_propagation(x, y, params, activation_function):
    # TODO learn the back propagation algorithm
    pass


def update(eta, params, activation_function):
    pass


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


def train(params, epochs, eta, activation_function, training_set, validation_set):
    for e in xrange(epochs):
        sum_loss = 0.0
        np.random.shuffle(training_set)

        for x, y in training_set:
            forward_cache = forward_propagation(x, activation_function, params)
            loss = calculate_loss(y, forward_cache["y_hat"])
            sum_loss += loss
            gradients = back_propagation(x, y, params, activation_function)
            params = update(eta, params, gradients)

        validation_loss, accuracy = predict_on_validation(validation_set, params, activation_function)

        print e, sum_loss / training_set[0].shape[0], validation_loss, "{}%".format(accuracy * 100)


if __name__ == "__main__":
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

    train(params, epochs, eta, sigmoid, training_set, validation_set)
