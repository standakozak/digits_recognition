import cProfile
from networks.neural_network_2 import NeuralNetwork, CrossEntropyCost, MeanSquaredErrorCost, load_network
from mnist_loader import load_fashion, load_mnist, load_doodles

from numpy.random import seed

if __name__ == "__main__":
    #seed(123456789)
    digits_recognition_neural_network = NeuralNetwork([784, 30, 10], cost_function=CrossEntropyCost())
    digits_recognition_neural_network.save_network("trained_networks/untrained_cec.json")
    training_data, validation_data, test_data = load_doodles()

    eta = 0.5
    epochs = 10
    mini_batch_size = 10
    # cProfile.run(f"digits_recognition_neural_network.train_network(training_data, mini_batch_size={mini_batch_size}, learning_rate={eta}, test_data=test_data, tests=10000, epochs={epochs})")
    digits_recognition_neural_network.train_network(training_data, mini_batch_size=mini_batch_size, learning_rate=eta, test_data=test_data, tests=10000, epochs=epochs)
    digits_recognition_neural_network.save_network("trained_networks/doodles_cec.json")
