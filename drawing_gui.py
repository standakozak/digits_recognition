import tkinter as tk
from gui.CanvasDrawing import CanvasDrawing
from gui.MainScreen import MainScreen
from gui.ViewProgress import ViewProgress
from networks.neural_network_2 import NeuralNetwork, activation_function, MeanSquaredErrorCost, CrossEntropyCost

import threading


class NeuralNetworksGUI(tk.Tk):
    FRAMES = (MainScreen, CanvasDrawing, ViewProgress)

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

    def update_network(self, net):
        self.network = net
        self.network_created = True
        self.current_frame.update_buttons(self.network_created, self.training_running)

    def show_frame(self, frame_obj):
        self.current_frame = self.frames_dict[frame_obj]
        print(f"Opening frame {self.current_frame}")
        self.current_frame.tkraise()

    def train_one_epoch(self):
        if self.epochs_to_run > 0:
            self.network.train_network(
                self.training_data, mini_batch_size=self.mini_batch_size, learning_rate=self.learning_rate, 
                test_data=self.test_data, tests=10000, epochs=1, regularization=self.regularization
            ) 
            self.epochs_to_run -= 1

            self.set_training_running(False)

    def train_more_epochs(self):
        if self.epochs_to_run > 0:
            self.network.train_network(
                self.training_data, mini_batch_size=self.mini_batch_size, learning_rate=self.learning_rate, 
                test_data=self.test_data, tests=10000, epochs=self.epochs_to_run, regularization=self.regularization
            )
            self.epochs_to_run = 0
            self.set_training_running(False)
        else:
            self.set_training_running(False)

    def continue_training(self):
        self.set_training_running(True)
        threading.Thread(target=self.train_one_epoch).start()

    def set_training_running(self, val):
        self.training_running = val
        self.current_frame.update_buttons(self.network_created, self.training_running, self.epochs_to_run)

    def train_network(self, dataset_function, mini_batch_size, learning_rate, regularization, epochs, stop):
        self.training_data, self.validation_data, self.test_data = dataset_function()
        self.mini_batch_size = mini_batch_size
        self.learning_rate = learning_rate
        self.regularization = regularization

        self.epochs_to_run = epochs

        self.set_training_running(True)
        if stop:
            threading.Thread(target=self.train_one_epoch).start()
        else:
            threading.Thread(target=self.train_more_epochs).start()
                
if __name__ == "__main__":
    main_app = NeuralNetworksGUI()
    main_app.mainloop()
