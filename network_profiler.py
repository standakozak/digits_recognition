import cProfile
from neural_network_2 import NeuralNetwork, CrossEntropyCost

from book_code.neural_network_2 import Network
from book_code.mnist_loader import load_data_wrapper

from numpy.random import seed

if __name__ == "__main__":
    seed(123456789)
    # Testing my network
    digits_recognition_neural_network = NeuralNetwork([784, 30, 10], cost_function=CrossEntropyCost())
    training_data, validation_data, test_data = load_data_wrapper()

    eta = 0.5
    epochs = 10
    mini_batch_size = 10
    cProfile.run(f"digits_recognition_neural_network.train_network(training_data, mini_batch_size={mini_batch_size}, learning_rate={eta}, test_data=test_data, tests=10000, epochs={epochs})")
    #digits_recognition_neural_network.train_network(training_data, mini_batch_size=10, learning_rate=3, test_data=test_data, tests=10000, epochs=10)

    # Testing original network
    net = Network([784, 30, 10], cost=CrossEntropyCost)
    #net.large_weight_initializer()
    #net.SGD(training_data, 10, 100, 0.5, evaluation_data=test_data, monitor_evaluation_accuracy=True)
    
    # #cProfile.run("digits_recognition_neural_network.train_network(training_data, mini_batch_size=1000, learning_rate=2, test_data=test_data, tests=1000)")
    # digits_recognition_neural_network.SGD(training_data, epochs=30, mini_batch_size=10, eta=3, test_data=test_data, epochs=30)
