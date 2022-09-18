import cProfile
from digits_recognition import NeuralNetwork
from original_digits_recognition import Network
from mnist_loader import load_data_wrapper

if __name__ == "__main__":
    # Testing my network
    digits_recognition_neural_network = NeuralNetwork([784, 30, 10])
    training_data, validation_data, test_data = load_data_wrapper()
    #cProfile.run("digits_recognition_neural_network.train_network(training_data, mini_batch_size=1000, learning_rate=2, test_data=test_data, tests=1000)")
    digits_recognition_neural_network.train_network(training_data, mini_batch_size=100, learning_rate=0.2, test_data=test_data, tests=10000, epochs=10)

    # Testing original network
    digits_recognition_neural_network = Network([784, 30, 10])
    # training_data, validation_data, test_data = load_data_wrapper()
    # #cProfile.run("digits_recognition_neural_network.train_network(training_data, mini_batch_size=1000, learning_rate=2, test_data=test_data, tests=1000)")
    # digits_recognition_neural_network.SGD(training_data, epochs=30, mini_batch_size=10, eta=3, test_data=test_data, epochs=30)
