import math
import numpy as np
import book_code.mnist_loader as mnist_loader


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
                new_biases = [np.random.random() for _ in range(layer_nodes)]
                new_layer = Layer(weights=new_weights, biases=new_biases)
                self.layers.append(new_layer)

                inputs = layer_nodes  # Current number of nodes becomes the number of inputs for the next layer

    def process_input(self, input_object):
        current_output = input_object
        
        for layer in self.layers:
            current_output = layer.calculate_outputs(current_output)
        return current_output

    def __repr__(self) -> str:
        return_string = f"Neural network with {len(self.layers)} layer(s)\n"
        for layer_num, layer_object in enumerate(self.layers):
            return_string += f"{layer_num + 1}) {repr(layer_object)}\n"
        return return_string


    def calculate_cost_of_one_input(self, input_object, desired_outputs) -> int:
        one_input_cost = 0

        real_outputs = self.process_input(input_object)
        for real_output, desired_output in zip(real_outputs, desired_outputs):
            one_input_cost += math.pow((desired_output - real_output), 2)
        
        return one_input_cost
    
    def cost_of_multiple_inputs(self, inputs_list, desired_outputs) -> int:
        total_cost = 0
        num_of_inputs = len(inputs_list)
        for input_object, output_object in zip(inputs_list, desired_outputs):
            total_cost += self.calculate_cost_of_one_input(input_object, output_object)

        return (total_cost / num_of_inputs)


    def apply_gradients(self, gradient_w, gradient_b, eta):
        delta_w = [weights * (-eta) for weights in gradient_w]
        delta_b = [biases * (-eta) for biases in gradient_b]
        ## Apply weight gradient
        for layer_num, layer in enumerate(self.layers):
            layer.weights += delta_w[layer_num]
            layer.biases += delta_b[layer_num]


    def train_network(self, training_data, mini_batch_size=100, learning_rate=3.0, test_data=None):
        h = 0.01

        mini_batches = make_mini_batches(training_data, mini_batch_size)

        for mini_batch in mini_batches:
            inputs = [data_point[0] for data_point in mini_batch]
            outputs = [data_point[1] for data_point in mini_batch]

            original_cost = self.cost_of_multiple_inputs(inputs, outputs)
            
            print(f"Cost: {original_cost}")

            # Create gradient lists
            gradient_w = [np.zeros(layer.weights.shape) for layer in self.layers]
            gradient_b = [np.zeros(layer.biases.shape) for layer in self.layers]

            for layer_index, layer in enumerate(self.layers):
                for node_in in range(len(layer.weights)):
                    for layer_node_index in range(len(layer.weights[node_in])):

                        layer.weights[node_in][layer_node_index] += h
                        weight_derivative = (self.cost_of_multiple_inputs(inputs, outputs) - original_cost) / h
                        layer.weights[node_in][layer_node_index] -= h
                        gradient_w[layer_index][node_in][layer_node_index] = weight_derivative


            for layer_index, layer in enumerate(self.layers):
                for layer_node_index in range(len(layer.biases)):
                    layer.biases[layer_node_index] += h
                    bias_derivative = (self.cost_of_multiple_inputs(inputs, outputs) - original_cost) / h
                    layer.biases[layer_node_index] -= h

                    gradient_b[layer_index][layer_node_index] = bias_derivative

            self.apply_gradients(gradient_w, gradient_b, learning_rate)


class Layer:
    def __init__(self, weights, biases) -> None:
        # Sets the layer's weights and biases
        # Parameters: weights: list of lists of ints ([1, 1], [1, 1], [1, 1]) for layer of three nodes and two inputs
        #             biases: list of ints

        self.weights = np.asarray(weights)
        self.biases = np.asarray(biases)

    def calculate_outputs(self, input_object):
        outputs = []
        for node_index, bias in enumerate(self.biases):
            current_node_output = np.dot(self.weights[node_index], input_object) + bias

            # Calling activation function
            outputs.append(activation_function(current_node_output))

        return outputs

    def test_input_list(self, inputs_list):
        outputs = []
        for input_object in inputs_list:
            current_output = self.calculate_outputs(input_object)
            outputs.append(current_output)

        return outputs

    def __repr__(self) -> str:
        return_string = f"Weights: {self.weights}, biases: {self.biases}"
        return return_string


def activation_function(weighed_input):
    ## Sigmoid function
    output = 1 / (1 + math.pow(math.e, -weighed_input))
    return output


def make_mini_batches(data, mini_batch_size):
    np.random.shuffle(data)
    mini_batches = [data[index:index+mini_batch_size] for index in range(0, len(data), mini_batch_size)]
    return mini_batches


digits_recognition_neural_network = NeuralNetwork([784, 10, 10])
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
digits_recognition_neural_network.train_network(training_data, mini_batch_size=10)
