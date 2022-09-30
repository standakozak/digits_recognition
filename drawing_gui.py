import tkinter as tk
from gui.CanvasDrawing import CanvasDrawing
from gui.MainScreen import MainScreen
from gui.ViewProgress import ViewProgress
from networks.neural_network_2 import NeuralNetwork, activation_function, MeanSquaredErrorCost, CrossEntropyCost

import threading


class NeuralNetworksGUI(tk.Tk):
    FRAMES = (MainScreen, CanvasDrawing, ViewProgress)
    IMAGE_RESOLUTION = 28

    def __init__(self) -> None:
        super().__init__()
        self.network = None
        self.network_created = False
        self.training_running = False

        self.training_data = None
        self.validation_data = None
        self.test_data = None
        self.mini_batch_size = None
        self.learning_rate = None
        self.regularization = None
        self.epochs_to_run = 0

        self.last_test_answers = []
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

    def update_network(self, net):
        self.network = net
        self.network_created = True
        self.current_frame.update_elements()

    def test_drawn_image(self, input_object):
        input_resized = input_object.reshape((self.IMAGE_RESOLUTION ** 2, 1))
        probabilities = self.network.output_probabilities(input_resized) * 100

        print("Guessing drawn image:")
        for index, certainty in enumerate(probabilities):
            print(f"{index}: {round(float(certainty), 3)} %")
    
    def test_network(self):
        correct, total_inputs, test_cost, self.last_test_answers = self.network.test_network(self.test_data, monitor_cost=True)
        self.last_test_accuracies.append(correct / total_inputs)
        self.last_test_costs.append(test_cost)
        print(f"Test: ({correct} / {total_inputs})   {(correct * 100) / total_inputs} %")

    def train_network(self, stop):
        self.set_training_running(True)
        while self.training_running:
            self.network.train_network(
                self.training_data, mini_batch_size=self.mini_batch_size, learning_rate=self.learning_rate, 
                test_data=None, tests=10000, epochs=1, regularization=self.regularization, monitor_accuracy=True
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


    def initialize_training(self, dataset_function, mini_batch_size, learning_rate, regularization, epochs, stop):
        self.last_test_accuracies = []
        self.last_test_costs = []
        self.last_training_accuracies = []
        self.last_training_costs = []
        
        self.training_data, self.validation_data, self.test_data = dataset_function()
        self.mini_batch_size = mini_batch_size
        self.learning_rate = learning_rate
        self.regularization = regularization

        self.epochs_to_run = epochs
        threading.Thread(target=lambda: self.start_training(stop)).start()

                
if __name__ == "__main__":
    main_app = NeuralNetworksGUI()
    main_app.mainloop()
