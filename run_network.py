import cProfile
from networks.neural_network_3 import NeuralNetwork, CrossEntropyCost, MeanSquaredErrorCost, load_network, SigmoidActivationFunction, run_with_early_stopping
from mnist_loader import load_fashion, load_mnist, load_doodles

from numpy.random import seed

if __name__ == "__main__":
    seed(54321)
    digits_recognition_neural_network = NeuralNetwork([784, 30, 10], cost_function=CrossEntropyCost(), activation_function=SigmoidActivationFunction())

    with cProfile.Profile() as pr:
        training_data, validation_data, test_data = load_mnist()
        
        #pr.print_stats()

    eta = 0.5  # 0.5 for CEC, 3.0 for MSE
    epochs = 50
    mini_batch_size = 10
    lmbda = 5.0  # 0.1 for 1000 training examples, 5.0 for the whole set of 50 000 examples

    with cProfile.Profile() as pr2:
        digits_recognition_neural_network.train_network(training_data[:1000], validation_data=test_data, mini_batch_size=mini_batch_size, learning_rate=eta, epochs=epochs, regularization=lmbda)
        
        #digits_recognition_neural_network.train_network(training_data, validation_data=test_data, mini_batch_size=mini_batch_size, learning_rate=eta, epochs=epochs, regularization=lmbda)
        
        #best_digits_recognition = run_with_early_stopping(digits_recognition_neural_network,
        #    training_data[:10000], validation_data, validations=1000, stopping_treshold=10,
        #    mini_batch_size=mini_batch_size, learning_rate=eta, regularization=lmbda
        #)
        #correct, total, _, _ = best_digits_recognition.test_network(test_data)
        #print(f"Final test accuracy: {correct/total*100}%")
        #pr2.print_stats()
