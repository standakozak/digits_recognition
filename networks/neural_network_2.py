import json
import math
import sys
from abc import ABC, abstractmethod, abstractstaticmethod
from typing import Optional, Union

import numpy as np
from mnist_loader import load_mnist
from scipy.special import expit


class ActivationFunction(ABC):
    @abstractstaticmethod
    def activation_function(weighed_inputs: np.ndarray) -> np.ndarray:
        """Calculates activations of weighed inputs"""
    
    @abstractmethod
    def activation_derivative(weighed_inputs: np.ndarray) -> np.ndarray:
        """Calculates the derivative of activation function"""


class SigmoidActivationFunction(ActivationFunction):
    @staticmethod
    def activation_function(weighed_inputs:np.ndarray) -> np.ndarray:
        ## Sigmoid function
        outputs = expit(weighed_inputs)
        return outputs


    def activation_derivative(self, weighed_inputs:np.ndarray) -> np.ndarray:
        ## Derivative of the sigmoid function
        ###  da
        ### ----
        ###  dz
        activation_values = self.activation_function(weighed_inputs)
        return activation_values * (1-activation_values)


class CostFunction(ABC):
    @abstractstaticmethod
    def calculate_cost(activations: np.ndarray, desired_outputs: np.ndarray) -> float:
        """Calculates cost of one input"""
    
    @abstractstaticmethod
    def cost_derivative(activations: np.ndarray, desired_outputs: np.ndarray) -> np.ndarray:
        """Calculates cost function derivative derivative of one input"""
    
    @abstractmethod
    def calculate_node_values(
        self, activations: np.ndarray, desired_outputs: np.ndarray, weighed_inputs: np.ndarray) -> np.ndarray:
        """Calculates output node values (node loss)"""

class CrossEntropyCost(CostFunction):
    @staticmethod
    def calculate_cost(activations, desired_outputs):
        one_input_cost = float(np.sum(
            np.nan_to_num(
                -desired_outputs * np.log(activations) - (1 - desired_outputs) * np.log(1 - activations)
            )
        ))
        
        return one_input_cost
    
    @staticmethod
    def cost_derivative(activations, desired_outputs):
        """
        This method is not called - the formula to calculate the node values is simplified
        by calculate_node_values method of this cost function
        """
        return (desired_outputs/activations) - ((1-desired_outputs)/(1-activations))

    def calculate_node_values(self, activations, desired_outputs, weighed_inputs, activation_func:ActivationFunction = SigmoidActivationFunction()) -> np.ndarray:
        return activations - desired_outputs


class MeanSquaredErrorCost(CostFunction):
    @staticmethod
    def calculate_cost(activations, desired_outputs):
        one_input_cost = 0.0
        for real_output, desired_output in zip(activations, desired_outputs):
            one_input_cost += math.pow((desired_output - real_output), 2) 
        return one_input_cost

    @staticmethod
    def cost_derivative(activations, desired_outputs) -> np.ndarray:
        return 2 * (activations - desired_outputs)
    
    def calculate_node_values(self, activations, desired_outputs, weighed_inputs, activation_func:ActivationFunction = SigmoidActivationFunction()):
        cost_to_activation_derivatives = self.cost_derivative(activations, desired_outputs)
        activation_to_weighed_input_der = activation_func.activation_derivative(weighed_inputs)
        node_values = cost_to_activation_derivatives * activation_to_weighed_input_der
        return node_values


def make_mini_batches(data:list[tuple[np.ndarray, np.ndarray]], mini_batch_size:int) -> list[list[tuple[np.ndarray, np.ndarray]]]:
    np.random.shuffle(data)
    mini_batches = [data[index:index+mini_batch_size] for index in range(0, len(data), mini_batch_size)]
    return mini_batches


def unvectorize_output(outputs: Union[float, np.ndarray]) -> float:
    if not np.isscalar(outputs):
        return outputs.argmax()
    return outputs


def classify_output(real_outputs:np.ndarray, desired_outputs:Union[float, np.ndarray], treshold=0) -> bool:
    """Checks if the highest output activation matches the desired output (and it exceeds a threshold))"""
    desired_output = unvectorize_output(desired_outputs)
    real_result = real_outputs.argmax()
    if real_result == desired_output and max(real_outputs) > treshold:
        return True
    return False


def load_network(file_name:str):
    with open(file_name, "r") as file:
        data = json.load(file)
    cost_function = getattr(sys.modules[__name__], data["cost"])
    cost = cost_function()
    activation_function = getattr(sys.modules[__name__], data["activation"])
    activation = activation_function()
    layers = []
    for layer_weights, layer_biases in zip(data["weights"], data["biases"]):
        new_layer = Layer(layer_weights, layer_biases, cost, activation)
        layers.append(new_layer)
    net = NeuralNetwork(sizes=data["sizes"], layers=layers, cost_function=cost, activation_function=activation)
    return net


class Layer:
    def __init__(self, weights:list[list[float]], biases:list[float], cost_function:CostFunction, activation_func:ActivationFunction) -> None:
        # Sets the layer's weights and biases
        # Parameters: weights: list of lists of ints ([1, 1], [1, 1], [1, 1]) for layer of three nodes and two inputs
        #             (shape = layer nodes(outputs), input_nodes)
        #             biases: list of floats

        self.weights = np.asarray(weights)
        self.biases = np.asarray(biases)
        self.node_values: np.ndarray = np.asarray([])

        self.activations: np.ndarray = np.zeros(self.biases.shape)  # Stores the activation values (outputs of this layer)
        self.weighed_inputs: np.ndarray = np.zeros(self.biases.shape)  # Stores outputs of this layer before going through the activation function

        self.cost_function = cost_function
        self.activation_function = activation_func

    def calculate_outputs(self, previous_activations:np.ndarray) -> np.ndarray:
        ## Calculate weighed inputs of this layer (dot product of weights, previous activations + bias)
        layer_weighed_inputs: np.ndarray = np.dot(self.weights, previous_activations) + self.biases
        layer_activations = self.activation_function.activation_function(layer_weighed_inputs)
        self.weighed_inputs = layer_weighed_inputs
        self.activations = layer_activations
        return layer_activations

    def calculate_output_node_values(self, expected_outputs:np.ndarray) -> None:
        node_values = []
        for weighed_input, expected_output, activation in zip(self.weighed_inputs, expected_outputs, self.activations):
            node_value = self.cost_function.calculate_node_values(activation, expected_output, weighed_input, self.activation_function)
            node_values.append(node_value)

        self.node_values = np.asarray(node_values)

    def calculate_hidden_node_values(self, next_layer) -> None:
        next_node_values = next_layer.node_values
        current_node_values = []
        activation_derivatives = self.activation_function.activation_derivative(self.weighed_inputs)
        current_node_values = np.dot(next_layer.weights.T, next_node_values) * activation_derivatives

        self.node_values = np.asarray(current_node_values)


    def calculate_layer_gradients(self, expected_outputs:np.ndarray, next_layer, previous_activations:np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if next_layer == None:
            self.calculate_output_node_values(expected_outputs)
        else:
            self.calculate_hidden_node_values(next_layer)

        ## Multiplies each node value with each activation from the previous layer
        layer_gradient_w = np.outer(self.node_values, previous_activations)
        layer_gradient_b = self.node_values

        return layer_gradient_w, layer_gradient_b

    def cost_derivative(self, activation:np.ndarray, desired_output:np.ndarray) -> np.ndarray:
        ###  dC
        ### ----
        ###  da 
        ### Partial derivative of the cost function and the real output

        return 2 * (activation - desired_output)


    def __repr__(self) -> str:
        return_string = f"Weights: {self.weights}, biases: {self.biases}"
        return return_string


class NeuralNetwork:
    def __init__(self, sizes: Optional[list[int]] = None, layers: Optional[list[Layer]] = None, 
        cost_function: CostFunction = CrossEntropyCost(), activation_function: ActivationFunction = SigmoidActivationFunction()
    ) -> None:
        self.cost_function = cost_function
        self.activation_function = activation_function
        self.sizes = sizes

        self.last_training_cost = 0.0
        self.last_training_accuracy = 0.0
        self.last_test_outputs = {}
        self.last_test_cost = 0.0
        self.last_test_accuracy = 0.0

        if layers is not None:
            self.layers = layers
        else:
            self.layers = []
            inputs = sizes[0]
            for layer_nodes in sizes[1:]:
                ## Creating a new layer
                new_weights = np.random.randn(layer_nodes, inputs)
                new_biases = list(np.random.randn(layer_nodes, 1))

                new_layer = Layer(weights=new_weights, biases=new_biases, 
                    cost_function=self.cost_function, activation_func=self.activation_function
                )
                self.layers.append(new_layer)

                inputs = layer_nodes  # Current number of nodes becomes the number of inputs for the next layer

    def __repr__(self) -> str:
        return_string = f"Neural network with {len(self.layers)} layer(s) and an input layer.\n"
        for layer_num, layer_object in enumerate(self.layers):
            return_string += f"{layer_num + 1}) {repr(layer_object)}\n"
        return return_string

    def save_network(self, file_name: str) -> None:
        weights = [layer.weights.tolist() for layer in self.layers]
        biases = [layer.biases.tolist() for layer in self.layers]
        data = {
            "sizes": self.sizes,
            "cost": self.cost_function.__class__.__name__,
            "activation": self.activation_function.__class__.__name__,
            "weights": weights,
            "biases": biases
        }
        with open(file_name, "w") as file:
            json.dump(data, file)

    def test_network(
        self, test_data: list[tuple[np.ndarray, np.ndarray]], num_of_datapoints: Optional[int] = None, monitor_cost=False
    ) -> tuple[int, int, float, list[tuple[np.ndarray, np.ndarray, np.ndarray, bool]]]:
        
        test_cost = 0.0
        np.random.shuffle(test_data)
        if num_of_datapoints is None:
            num_of_datapoints = len(test_data)

        test_data = test_data[:num_of_datapoints]
        
        correct_answers_num = 0
        answers = []
        for data_point in test_data:
            inputs:np.ndarray = data_point[0]
            desired_outputs:np.ndarray = data_point[1]
            real_outputs = self.process_input(inputs)

            answer = classify_output(real_outputs, desired_outputs, treshold=0)
            correct_answers_num += int(answer)
            answers.append((inputs, desired_outputs, real_outputs/sum(real_outputs), answer))

            if monitor_cost:
                test_cost += self.cost_function.calculate_cost(real_outputs, desired_outputs)
        return correct_answers_num, num_of_datapoints, test_cost/num_of_datapoints, answers

    def train_network(
            self, training_data: list[tuple[np.ndarray, np.ndarray]], mini_batch_size=10, learning_rate=0.05, 
            test_data: Optional[list[tuple[np.ndarray, np.ndarray]]] = None, tests: Optional[int] = None,
            epochs=1, regularization=0, monitor_accuracy=False
        ):
        total_inputs = len(training_data)
        total_correct = 0
        total_cost: float = 0.0
        for epoch_num in range(epochs):
            epoch_cost = 0
            mini_batches = make_mini_batches(training_data, mini_batch_size)
            epoch_correct_answers = 0

            for mini_batch in mini_batches:
                delta_gradient_w = [np.zeros(layer.weights.shape) for layer in self.layers]
                delta_gradient_b = [np.zeros(layer.biases.shape) for layer in self.layers]

                for datapoint_input, datapoint_output in mini_batch:
                    delta_gradient_w, delta_gradient_b, cost = self.update_gradients(datapoint_input, datapoint_output, delta_gradient_w, delta_gradient_b)
                    
                    if monitor_accuracy:
                        activations = self.process_input(datapoint_input)
                        epoch_correct_answers += int(classify_output(activations, datapoint_output))

                self.apply_gradients(delta_gradient_w, delta_gradient_b, learning_rate/mini_batch_size)
                epoch_cost += cost

            total_cost += epoch_cost
            total_correct += epoch_correct_answers
            # Testing after each epoch
            if test_data is not None:
                print(f"{epoch_num+1}th epoch completed ({(epoch_num + 1)}/{epochs})")
                print(f"Average epoch cost: {epoch_cost/total_inputs}")
                correct, total, _, _ = self.test_network(test_data, tests)
                print(f"Test: ({correct} / {total})   {(correct * 100) / total} %")
        
        self.last_training_cost = total_cost / (epochs * total_inputs)
        self.last_training_accuracy = total_correct / (epochs * total_inputs)

    def update_gradients(self, dp_input: np.ndarray, expected_output: np.ndarray, gradient_w: np.ndarray, gradient_b: np.ndarray):
        """
        Calls the feedforward algorithm and then the backpropagation
        """
        one_input_cost = self.calculate_cost_of_one_input(dp_input, expected_output)

        old_layer = None
        ## next layer: layers[i+1]; previous layer: layers[i-1]
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

    def apply_gradients(self, gradient_w: np.ndarray, gradient_b: np.ndarray, eta: float):
        delta_w = [weights * (-eta) for weights in gradient_w]
        delta_b = [biases * (-eta) for biases in gradient_b]
        ## Apply weight gradient
        for layer_num, layer in enumerate(self.layers):
            layer.weights = layer.weights + delta_w[layer_num]
            layer.biases = layer.biases + delta_b[layer_num]

    def calculate_cost_of_one_input(self, dp_input: np.ndarray, desired_output: np.ndarray) -> float:
        outputs = self.process_input(dp_input)
        return self.cost_function.calculate_cost(outputs, desired_output)

    def process_input(self, input_object: np.ndarray):
        """
        Feedforward algorithm -> goes through the network, updates weighed inputs and activations of each layer
        Return activations of the last layer - network output
        """
        current_activations = input_object
        
        for layer in self.layers:
            current_activations = layer.calculate_outputs(current_activations)
        return current_activations
    
    def softmax_output(self, input_object: np.ndarray):
        activations = self.process_input(input_object)
        e_to_activations = np.power(math.e, activations)
        softmax_activations = e_to_activations / sum(e_to_activations) 
        return softmax_activations

    def output_probabilities(self, input_object: np.ndarray):
        activations = self.process_input(input_object)
        probabilities = activations / sum(activations) 
        return probabilities

if __name__ == "__main__":
    digits_network = NeuralNetwork([784, 30, 10], cost_function=CrossEntropyCost())
    training_data, validation_data, test_data = load_mnist()
    digits_network.train_network(training_data, mini_batch_size=10, learning_rate=0.5, test_data=test_data, tests=10000, epochs=10)
