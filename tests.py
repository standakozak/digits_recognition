from neural_network_1 import Layer, NeuralNetwork, activation_function, activation_function_from_array


def test_calculate_outputs():
    new_layer = Layer(weights=[[-10, -0.5, 8], [0.5, -1.5, 2]], biases=[-1.5, 2])
    inputs = [10, -20, 3]
    weighed_inputs = [-67.5, 43]
    activations = activation_function_from_array(weighed_inputs)
    
    layer_outputs = new_layer.calculate_outputs(inputs)

    assert all(new_layer.weighed_inputs == weighed_inputs)
    assert all(layer_outputs == activations)
    assert all(new_layer.activations == activations)


if __name__ == "__main__":
    test_calculate_outputs()