import cProfile
from neural_network_1 import NeuralNetwork
from book_code.original_digits_recognition import Network
from book_code.mnist_loader import load_data_wrapper

if __name__ == "__main__":
    # Testing my network
    digits_recognition_neural_network = NeuralNetwork([784, 30, 10])
    training_data, validation_data, test_data = load_data_wrapper()
    #cProfile.run("digits_recognition_neural_network.train_network(training_data, mini_batch_size=1000, learning_rate=2, test_data=test_data, tests=1000)")
    digits_recognition_neural_network.train_network(training_data, mini_batch_size=10, learning_rate=3, test_data=test_data, tests=10000, epochs=10)

    # Testing original network
    digits_recognition_neural_network = Network([784, 30, 10])
    # training_data, validation_data, test_data = load_data_wrapper()
    # #cProfile.run("digits_recognition_neural_network.train_network(training_data, mini_batch_size=1000, learning_rate=2, test_data=test_data, tests=1000)")
    # digits_recognition_neural_network.SGD(training_data, epochs=30, mini_batch_size=10, eta=3, test_data=test_data, epochs=30)
