import math
import numpy as np
import book_code.mnist_loader as mnist_loader
from scipy.special import expit



class NeuralNetwork:
    def __init__(self, sizes=None, layers=None) -> None:
        if layers is not None:
            self.layers = layers
        else:
            self.layers = []
            inputs = sizes[0]
            for layer_nodes in sizes[1:]:
                ## Creating a new layer
                new_weights = np.random.randn(layer_nodes, inputs)
                new_biases = list(np.random.randn(layer_nodes, 1))

                new_layer = Layer(weights=new_weights, biases=new_biases)
                self.layers.append(new_layer)

                inputs = layer_nodes  # Current number of nodes becomes the number of inputs for the next layer

    def process_input(self, input_object):
        """
        Feedforward algorithm -> goes through the network, updates weighed inputs and activations of each layer
        Return activations of the last layer - network output
        """
        current_activations = input_object
        
        for layer in self.layers:
            current_activations = layer.calculate_outputs(current_activations)
        return current_activations

    def __repr__(self) -> str:
        return_string = f"Neural network with {len(self.layers)} layer(s)\n"
        for layer_num, layer_object in enumerate(self.layers):
            return_string += f"{layer_num + 1}) {repr(layer_object)}\n"
        return return_string


    def calculate_cost_of_one_input(self, dp_input, desired_output) -> int:
        one_input_cost = 0

        real_outputs = self.process_input(dp_input)
        for real_output, desired_output in zip(real_outputs, desired_output):
            one_input_cost += math.pow((desired_output - real_output), 2)
        
        return one_input_cost

    def apply_gradients(self, gradient_w, gradient_b, eta):
        delta_w = [weights * (-eta) for weights in gradient_w]
        delta_b = [biases * (-eta) for biases in gradient_b]
        ## Apply weight gradient
        for layer_num, layer in enumerate(self.layers):
            layer.weights = layer.weights + delta_w[layer_num]
            layer.biases = layer.biases + delta_b[layer_num]

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


class Layer:
    def __init__(self, weights, biases) -> None:
        # Sets the layer's weights and biases
        # Parameters: weights: list of lists of ints ([1, 1], [1, 1], [1, 1]) for layer of three nodes and two inputs
        #             (In other words: shape of - layer nodes(outputs), input_nodes)
        #             biases: list of ints

        self.weights = np.asarray(weights)
        self.biases = np.asarray(biases)
        self.node_values = []

        self.activations = np.zeros(self.biases.shape)  # Stores the activation values (outputs of this layer)
        self.weighed_inputs = np.zeros(self.biases.shape)  # Stores outputs of this layer before going through the activation function

    def calculate_outputs(self, input_object):
        ## Calculate weighed inputs of this layer (dot product of weights, previous activations + bias)
        layer_weighed_inputs = np.asarray([(np.dot(self.weights[node_index], input_object) + bias) for node_index, bias in enumerate(self.biases)])
        layer_activations = activation_function_from_array(layer_weighed_inputs)
        self.weighed_inputs = layer_weighed_inputs
        self.activations = layer_activations
        return layer_activations

    def calculate_output_node_values(self, expected_outputs):
        node_values = []
        for weighed_input, expected_output, activation in zip(self.weighed_inputs, expected_outputs, self.activations):
            cost_to_activation_derivative = self.cost_derivative(activation, expected_output)
            activation_to_weighed_input_der = activation_derivative(weighed_input)
            node_values.append(cost_to_activation_derivative * activation_to_weighed_input_der)
        self.node_values = node_values

    def calculate_hidden_node_values(self, next_layer):
        next_node_values = next_layer.node_values
        current_node_values = []
        for node_index, current_node_weights in enumerate(next_layer.weights.T):
            activation_to_weighed_input_derivative = activation_derivative(self.weighed_inputs[node_index])
            current_node_values.append(np.dot(current_node_weights, next_node_values) * activation_to_weighed_input_derivative)

        self.node_values = current_node_values


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


def activation_function(weighed_input):
    ## Sigmoid function

    #output = 1 / (1 + pow(math.e, -weighed_input))
    return expit(weighed_input)


def activation_function_from_array(weighed_inputs):
    outputs = expit(weighed_inputs)
    return outputs


def activation_derivative(weighed_input):
    ## Derivative of the sigmoid function
    ###  da
    ### ----
    ###  dz
    activation_value = activation_function(weighed_input)
    return (activation_value * (1-activation_value))


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
    digits_recognition_neural_network.train_network(training_data, mini_batch_size=10, learning_rate=3, test_data=test_data, tests=200)
