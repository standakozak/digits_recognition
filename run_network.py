import cProfile
from networks.neural_network_2 import NeuralNetwork, CrossEntropyCost, MeanSquaredErrorCost, load_network, SigmoidActivationFunction
from mnist_loader import load_fashion, load_mnist, load_doodles

from numpy.random import seed

if __name__ == "__main__":
    seed(123456789)
    digits_recognition_neural_network = NeuralNetwork([784, 30, 10], cost_function=CrossEntropyCost(), activation_function=SigmoidActivationFunction())

    with cProfile.Profile() as pr:
        training_data, validation_data, test_data = load_mnist()
        
        #pr.print_stats()

    eta = 0.5
    epochs = 2
    mini_batch_size = 10

    with cProfile.Profile() as pr2:
        digits_recognition_neural_network.train_network(training_data, test_data=test_data, mini_batch_size=mini_batch_size, learning_rate=eta, epochs=epochs)
        pr2.print_stats()
