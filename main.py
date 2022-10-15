import tkinter as tk

from gui.CanvasDrawing import CanvasDrawing
from gui.MainScreen import MainScreen
from gui.ViewProgress import ViewProgress
from gui.BrowseOutputs import BrowseOutputs
from networks.neural_network_2 import NeuralNetwork, activation_function, MeanSquaredErrorCost, CrossEntropyCost, unvectorize_output
from mnist_loader import load_mnist, load_fashion, load_doodles

import threading

DOODLE_CATEGORIES = ["axe", "bicycle", "broom", "bucket", "candle", "chair", "eyeglasses", "guitar", "key", "ladder"]

class NeuralNetworksGUI(tk.Tk):
    FRAMES = (MainScreen, CanvasDrawing, ViewProgress, BrowseOutputs)
    IMAGE_RESOLUTION = 28
    datasets = {
        "MNIST": (
            load_mnist, ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
        ),
        "Fashion": (
            load_fashion, ["T-Shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
        ),
        "Doodles": (
            lambda: load_doodles(DOODLE_CATEGORIES), DOODLE_CATEGORIES
        )
    }

    def __init__(self) -> None:
        super().__init__()

        # For MainScreen - setting up the network
        self.network = None
        self.network_created = False
        self.training_running = False

        # For MainScreen - training the network
        self.training_data = None
        self.validation_data = None
        self.test_data = None
        self.mini_batch_size = None
        self.learning_rate = None
        self.regularization = None
        self.num_of_tests = None
        self.epochs_to_run = 0
        self.total_training_epochs = 0

        self.dataset = self.datasets["MNIST"]
        
        # For BrowseOutputs
        self.last_test_answers = []
        self.current_output_index = -1
        
        # For ViewProgress
        self.last_test_accuracies = []
        self.last_test_costs = []
        self.last_training_accuracies = []
        self.last_training_costs = []

        self.title("Neural Networks")
        self.geometry("890x590")
        self.configure(background='white')

        self.container = tk.Frame(self, width=890, height=590)
        self.container.pack(side="top", expand=True, fill="both")

        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)

        self.frames_dict = {}
        for frame_obj in self.FRAMES:
            frame = frame_obj(self.container, self)
            frame.grid(row=0, column=0, sticky="nsew")
            self.frames_dict[frame_obj.__name__] = frame
        
        self.show_frame("MainScreen")

    def show_frame(self, frame_name):
        self.current_frame = self.frames_dict[frame_name]
        print(f"Opening {frame_name}")
        self.current_frame.update_elements()
        self.current_frame.tkraise()

    ## BrowseOutputs Frame
    def show_next_test_output(self, incorrect=False):
        if not self.last_test_answers:
            return
        
        image = None
        image_found = False
        image_index = self.current_output_index
        while not image_found:
            image_index += 1
            
            if image_index >= len(self.last_test_answers):
                image_index = 0
                if self.current_output_index == -1:
                    image_found = True
            
            image = self.last_test_answers[image_index]
            if image[3] == False or not incorrect or image_index == self.current_output_index:
                image_found = True
                self.current_output_index = image_index
        
        inputs = image[0].reshape((self.IMAGE_RESOLUTION, self.IMAGE_RESOLUTION))
        correct_answer = self.dataset[1][unvectorize_output(image[1])]
        real_answers = self.match_probabilities_with_answers(image[2])

        self.current_frame.show_output(inputs, correct_answer, real_answers, image[3])

    ## CanvasDrawing Frame
    def test_drawn_image(self, input_object):
        input_resized = input_object.reshape((self.IMAGE_RESOLUTION ** 2, 1))
        probabilities = self.network.output_probabilities(input_resized)

        zipped_sorted = self.match_probabilities_with_answers(probabilities)
        if self.current_frame.__class__.__name__ == "CanvasDrawing":
            self.current_frame.display_probabilities(zipped_sorted)

    def match_probabilities_with_answers(self, activations):
        answers = self.dataset[1]
        zipped = list(zip(answers, activations * 100))
        sorted_answers_probabilities = sorted(zipped, key=lambda x: x[1], reverse=True)
        return sorted_answers_probabilities
        
    ## MainScreen Frame
    def update_network(self, net):
        self.network = net
        self.network_created = True
        self.current_frame.update_elements()

    def test_network(self):
        correct, total_inputs, test_cost, self.last_test_answers = self.network.test_network(self.test_data, num_of_datapoints=self.num_of_tests, monitor_cost=True)
        self.last_test_accuracies.append(correct / total_inputs)
        self.last_test_costs.append(test_cost)
        print(f"Test: ({correct} / {total_inputs})   {(correct * 100) / total_inputs} %")
    
    def train_network(self, stop):
        self.set_training_running(True)
        while self.training_running:
            self.network.train_network(
                self.training_data, mini_batch_size=self.mini_batch_size, learning_rate=self.learning_rate, 
                test_data=None, tests=0, epochs=1, regularization=self.regularization, monitor_accuracy=True
            )
            self.last_training_accuracies.append(self.network.last_training_accuracy)
            self.last_training_costs.append(self.network.last_training_cost)
            self.test_network()

            self.epochs_to_run -= 1
            self.current_frame.update_elements()
            if stop or self.epochs_to_run <= 0:
                self.set_training_running(False)

    def continue_training(self):
        threading.Thread(target=lambda: self.train_network(stop=True)).start()

    def set_training_running(self, val):
        self.training_running = val
        self.current_frame.update_elements()
            
    def start_training(self, stop):
        print(f"Initial test")
        self.test_network()
        self.train_network(stop=stop)
    
    def stop_training(self):
        self.training_running = False
        self.epochs_to_run = 0

    def initialize_training(self, dataset, mini_batch_size, learning_rate, regularization, epochs, stop, num_of_tests):
        self.dataset = self.datasets[dataset]
        self.last_test_accuracies = []
        self.last_test_costs = []
        self.last_training_accuracies = [None]
        self.last_training_costs = [None]
        
        self.training_data, self.validation_data, self.test_data = self.dataset[0]()
        self.mini_batch_size = mini_batch_size
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.num_of_tests = num_of_tests

        self.epochs_to_run = epochs
        self.total_training_epochs = epochs
        threading.Thread(target=lambda: self.start_training(stop)).start()

                
if __name__ == "__main__":
    main_app = NeuralNetworksGUI()
    main_app.mainloop()
