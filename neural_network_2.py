from argparse import ONE_OR_MORE
import math
import numpy as np
import book_code.mnist_loader as mnist_loader
from scipy.special import expit


class CrossEntropyCost:
    def __init__(self):
        pass

    @staticmethod
    def calculate_cost(activations, desired_outputs):
        y = np.asarray(desired_outputs)
        a = np.asarray(activations)
        one_input_cost = np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))
        # one_input_cost = 0
        # for real_output, desired_output in zip(real_outputs, desired_outputs):
        #     one_input_cost += np.nan_to_num(-desired_output * np.log(real_output) - (1-desired_output) * np.log(1-real_output))
        return one_input_cost
    
    @staticmethod
    def cost_derivative(activation, desired_output):
        """
        This method is not necessary - the formula to calculate the node values is simplified by calculate_node_values method
        """
        return (desired_output/activation) - ((1-desired_output)/(1-activation))

    def calculate_node_values(self, activations, desired_outputs, weighted_inputs):
        """
        Calculates the node values of the output layer for computing the gradient vectors
        """
        return activations - desired_outputs


class MeanSquaredErrorCost:
    def __init__(self) -> None:
        pass

    @staticmethod
    def calculate_cost(activations, desired_outputs):
        one_input_cost = 0
        for real_output, desired_output in zip(activations, desired_outputs):
            one_input_cost += math.pow((desired_output - real_output), 2) 
        return one_input_cost

    @staticmethod
    def cost_derivative(activations, desired_outputs):
        return 2 * (activations - desired_outputs)
    
    def calculate_node_values(self, activations, desired_outputs, weighted_inputs):
        cost_to_activation_derivatives = self.cost_derivative(activations, desired_outputs)
        activation_to_weighted_input_der = activation_derivative(weighted_inputs)
        node_values = cost_to_activation_derivatives * activation_to_weighted_input_der
        return node_values


class NeuralNetwork(object):
    def __init__(self, sizes=None, layers=None, cost_function=CrossEntropyCost) -> None:
        self.cost_function = cost_function

        if layers is not None:
            self.layers = layers
        else:
            self.layers = []
            inputs = sizes[0]
            for layer_nodes in sizes[1:]:
                ## Creating a new layer
                new_weights = np.random.randn(layer_nodes, inputs)
                new_biases = list(np.random.randn(layer_nodes, 1))

                new_layer = Layer(weights=new_weights, biases=new_biases, cost_function=self.cost_function)
                self.layers.append(new_layer)

                inputs = layer_nodes  # Current number of nodes becomes the number of inputs for the next layer

    def __repr__(self) -> str:
        return_string = f"Neural network with {len(self.layers)} layer(s) and an input layer.\n"
        for layer_num, layer_object in enumerate(self.layers):
            return_string += f"{layer_num + 1}) {repr(layer_object)}\n"
        return return_string

    def test_network(self, test_data, num_of_datapoints=None):
        np.random.shuffle(test_data)
        if num_of_datapoints is None:
            num_of_datapoints = len(test_data)

        test_data = test_data[:num_of_datapoints]
        
        correct_answers_num = 0
        wrong_answers = []
        for data_point in test_data:
            inputs = data_point[0]
            desired_outputs = data_point[1]
            real_outputs = self.process_input(inputs)

            answer = classify_output(real_outputs, desired_outputs, certainty=0)
            if answer:
                correct_answers_num += 1
        return correct_answers_num, num_of_datapoints, wrong_answers

    def train_network(self, training_data, mini_batch_size=10, learning_rate=0.05, test_data=None, tests=None, epochs=1):
        total_inputs = len(training_data)
        
        if test_data is not None:
            print(f"Initial test")
            correct, total, wrong_cases = self.test_network(test_data, tests)
            print(f"Test: ({correct} / {total})   {(correct * 100) / total} %")

        for epoch_num in range(epochs):
            total_cost = 0
            mini_batches = make_mini_batches(training_data, mini_batch_size)

            for mini_batch_index, mini_batch in enumerate(mini_batches):
                delta_gradient_w = [np.zeros(layer.weights.shape) for layer in self.layers]
                delta_gradient_b = [np.zeros(layer.biases.shape) for layer in self.layers]

                for datapoint_input, datapoint_output in mini_batch:
                    delta_gradient_w, delta_gradient_b, cost = self.update_gradients(datapoint_input, datapoint_output, delta_gradient_w, delta_gradient_b)

                total_cost += cost

                ### Testing after every batch
                if test_data is not None and epochs == 1:
                    print(f"{mini_batch_index+1}th mini-batch completed ({(mini_batch_index + 1)*mini_batch_size}/{total_inputs})")
                    correct, total, wrong_cases = self.test_network(test_data, tests)
                    print(f"Test: ({correct} / {total})   {(correct * 100) / total} %")

                self.apply_gradients(delta_gradient_w, delta_gradient_b, learning_rate/mini_batch_size)
            
            if test_data is not None and epochs != 1:
                print(f"{epoch_num+1}th epoch completed ({(epoch_num + 1)}/{epochs})")
                print(f"Average cost: {total_cost / len(training_data)}")
                correct, total, wrong_cases = self.test_network(test_data, tests)
                print(f"Test: ({correct} / {total})   {(correct * 100) / total} %")

    def update_gradients(self, dp_input, expected_output, gradient_w, gradient_b):
        """
        Calls the feedforward algorithm and then the backpropagation
        """
        one_input_cost = self.calculate_cost_of_one_input(dp_input, expected_output)

        old_layer = None
        ## next layer = layer[i+1]; previous layer = layer[i-1]
        for layer_index, layer in reversed(list(enumerate(self.layers))):
            if layer_index > 0:
                previous_activations = self.layers[layer_index-1].activations
            else:
                previous_activations = dp_input

            layer_gradient_w, layer_gradient_b = layer.calculate_layer_gradients(expected_output, old_layer, previous_activations)
            
            gradient_w[layer_index] = gradient_w[layer_index] + layer_gradient_w
            gradient_b[layer_index] = gradient_b[layer_index] + layer_gradient_b

            old_layer = layer

        return gradient_w, gradient_b, one_input_cost

    def apply_gradients(self, gradient_w, gradient_b, eta):
        delta_w = [weights * (-eta) for weights in gradient_w]
        delta_b = [biases * (-eta) for biases in gradient_b]
        ## Apply weight gradient
        for layer_num, layer in enumerate(self.layers):
            layer.weights = layer.weights + delta_w[layer_num]
            layer.biases = layer.biases + delta_b[layer_num]

    def calculate_cost_of_one_input(self, dp_input, desired_output) -> int:
        outputs = self.process_input(dp_input)
        return self.cost_function.calculate_cost(outputs, desired_output)

    def process_input(self, input_object):
        """
        Feedforward algorithm -> goes through the network, updates weighted inputs and activations of each layer
        Return activations of the last layer - network output
        """
        current_activations = input_object
        
        for layer in self.layers:
            current_activations = layer.calculate_outputs(current_activations)
        return current_activations

class Layer:
    def __init__(self, weights, biases, cost_function) -> None:
        # Sets the layer's weights and biases
        # Parameters: weights: list of lists of ints ([1, 1], [1, 1], [1, 1]) for layer of three nodes and two inputs
        #             (In other words: shape of - layer nodes(outputs), input_nodes)
        #             biases: list of ints

        self.weights = np.asarray(weights)
        self.biases = np.asarray(biases)
        self.node_values = []

        self.activations = np.zeros(self.biases.shape)  # Stores the activation values (outputs of this layer)
        self.weighted_inputs = np.zeros(self.biases.shape)  # Stores outputs of this layer before going through the activation function

        self.cost_function = cost_function

    def calculate_outputs(self, previous_activations):
        ## Calculate weighted inputs of this layer (dot product of weights, previous activations + bias)
        #layer_weighted_inputs = np.asarray([(np.dot(self.weights[node_index], previous_activations) + bias) for node_index, bias in enumerate(self.biases)])
        layer_weighted_inputs = np.dot(self.weights, previous_activations) + self.biases
        layer_activations = activation_function(layer_weighted_inputs)
        self.weighted_inputs = layer_weighted_inputs
        self.activations = layer_activations
        return layer_activations

    def calculate_output_node_values(self, expected_outputs):
        node_values = []
        for weighted_input, expected_output, activation in zip(self.weighted_inputs, expected_outputs, self.activations):
            node_value = self.cost_function.calculate_node_values(activation, expected_output, weighted_input)
            node_values.append(node_value)

        self.node_values = np.asarray(node_values)

    def calculate_hidden_node_values(self, next_layer):
        next_node_values = next_layer.node_values
        current_node_values = []
        activation_derivatives = activation_derivative(self.weighted_inputs)
        current_node_values = np.dot(next_layer.weights.T, next_node_values) * activation_derivatives
        # for node_index, current_node_weights in enumerate(next_layer.weights.T):
        #     activation_to_weighted_input_derivative = activation_derivative(self.weighted_inputs[node_index])
        #     current_node_values.append(np.dot(current_node_weights, next_node_values) * activation_to_weighted_input_derivative)

        self.node_values = np.asarray(current_node_values)


    def calculate_layer_gradients(self, expected_outputs, next_layer, previous_activations):
        if next_layer == None:
            self.calculate_output_node_values(expected_outputs)
        else:
            self.calculate_hidden_node_values(next_layer)

        current_node_values = np.asarray(self.node_values)
        previous_activations = np.asarray(previous_activations)

        ## Multiplies each node value with each activation from the previous layer
        layer_gradient_w = np.outer(current_node_values, previous_activations)
        layer_gradient_b = current_node_values

        return layer_gradient_w, layer_gradient_b

    def cost_derivative(self, activation, desired_output):
        ###  dC
        ### ----
        ###  da 
        ### Partial derivative of the cost function and the real output

        #one_input_cost = 0
        #for real_output, desired_output in zip(real_outputs, desired_outputs):
        #    one_input_cost += 2 * (real_output - desired_output)
        #return one_input_cost
        return 2 * (activation - desired_output)


    def __repr__(self) -> str:
        return_string = f"Weights: {self.weights}, biases: {self.biases}"
        return return_string

def activation_function(weighted_inputs):
    ## Sigmoid function
    outputs = expit(weighted_inputs)
    return outputs


def activation_derivative(weighted_inputs):
    ## Derivative of the sigmoid function
    ###  da
    ### ----
    ###  dz
    activation_values = activation_function(weighted_inputs)
    return activation_values * (1-activation_values)


def make_mini_batches(data, mini_batch_size):
    np.random.shuffle(data)
    mini_batches = [data[index:index+mini_batch_size] for index in range(0, len(data), mini_batch_size)]
    return mini_batches


def classify_output(real_outputs, desired_output, certainty=0):
    real_result = real_outputs.argmax()
    if real_result == desired_output and max(real_outputs) > certainty:
        return True
    return False


if __name__ == "__main__":
    digits_recognition_neural_network = NeuralNetwork([784, 30, 10])
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    digits_recognition_neural_network.train_network(training_data, mini_batch_size=100, learning_rate=3, test_data=test_data, tests=2000, epochs=2)
