import tkinter as tk
from tkinter import W, ttk
from networks.neural_network_2 import NeuralNetwork, load_network, CrossEntropyCost, MeanSquaredErrorCost, activation_function
from mnist_loader import load_mnist, load_fashion

from tkinter import filedialog

class MainScreen(tk.Frame):
    dataset_functions = {
        "MNIST": load_mnist,
        "Fashion": load_fashion,
        "Doodles": load_mnist
    }
    activation_functions = {
        "Sigmoid": activation_function,
        "Softmax": activation_function
    }
    cost_functions = {
        "Mean Squared Error": MeanSquaredErrorCost,
        "Cross Entropy Cost": CrossEntropyCost
    }

    def __init__(self, container, controller) -> None:
        tk.Frame.__init__(self, container)
        self.controller = controller

        self.grid_rowconfigure(1, weight=1)
        self.grid_rowconfigure(4, weight=3)
        
        self.grid_rowconfigure((5,), weight=1)
        self.grid_columnconfigure((0), weight=1)
        
        tk.Label(self, text='Create Neural Network', font=('arial', 24, 'normal')).grid(row=0, column=0, columnspan=2, sticky="nsew")
        top_inputs_frame = tk.Frame(self, width=200)
        top_inputs_frame.grid(row=1, column=0, sticky="nsew", columnspan=2)
        top_inputs_frame.columnconfigure((0, 1), weight=1)

        top_buttons_frame = tk.Frame(self)
        top_buttons_frame.columnconfigure((0, 1), weight=1)
        top_buttons_frame.grid(row=2, column=0, columnspan=2, sticky="nsew")
        
        tk.Label(self, text='Train Neural Network', font=('arial', 24, 'normal')).grid(row=3, column=0, columnspan=2, sticky="nsew")

        bottom_inputs_frame = tk.Frame(self, width=200)
        bottom_inputs_frame.grid(row=4, column=0, sticky="nsew")
        bottom_inputs_frame.columnconfigure((0, 1), weight=1)

        bottom_buttons_frame = tk.Frame(self)
        bottom_buttons_frame.grid(row=5, column=0, columnspan=2, sticky="nsew")
        bottom_buttons_frame.columnconfigure((0, 1, 2, 3, 4, 5), weight=1)

        ## Top inputs frame
        # Sizes input field
        tk.Label(top_inputs_frame, text='Sizes:', font=('arial', 12, 'normal')).grid(row=0, column=0, sticky="e")
        self.sizes_input = tk.Entry(top_inputs_frame, font=('arial', 12, 'normal'))
        self.sizes_input.insert(1, "784, 30, 10")
        self.sizes_input.grid(row=0, column=1, sticky="w")

        # Activation function choice
        tk.Label(top_inputs_frame, text='Activation function:', font=('arial', 12, 'normal')).grid(row=1, column=0, sticky="e")

        self.activation_func_box = ttk.Combobox(top_inputs_frame, values=["Sigmoid", 'Softmax'], font=('arial', 12, 'normal'), width=12, state="readonly")
        self.activation_func_box.grid(row=1, column=1, sticky="w")
        self.activation_func_box.current(0)

        # Cost function choice
        tk.Label(top_inputs_frame, text='Cost function:', font=('arial', 12, 'normal')).grid(row=2, column=0, sticky="e")
        self.cost_func_box = ttk.Combobox(top_inputs_frame, values=['Mean Squared Error', 'Cross Entropy Cost'], font=('arial', 12, 'normal'), width=20, state="readonly")
        self.cost_func_box.grid(row=2, column=1, sticky="w")
        self.cost_func_box.current(0)

        ## Top buttons frame
        # 'Load' button
        self.load_button = tk.Button(top_buttons_frame, text='Load from file', font=('arial', 15, 'normal'), command=self.load_network)
        self.load_button.grid(row=0, column=0)
        # 'Create' button
        self.create_network_button = tk.Button(top_buttons_frame, text='Create Network', font=('arial', 15, 'normal'), command=self.create_network)
        self.create_network_button.grid(row=0, column=1)
        
        ## Bottom inputs frame
        # Dataset choice
        tk.Label(bottom_inputs_frame, text='Dataset:', font=('arial', 12, 'normal')).grid(row=0, column=0, sticky="e")
        self.dataset_box= ttk.Combobox(bottom_inputs_frame, values=['MNIST', 'Fashion', 'Doodles'], font=('arial', 12, 'normal'), width=15, state="readonly")
        self.dataset_box.grid(row=0, column=1, sticky="w")
        self.dataset_box.current(0)

        # Minibatch spinbox
        tk.Label(bottom_inputs_frame, text='Mini batch size:', font=('arial', 12, 'normal')).grid(row=1, column=0, sticky="e")
        self.mini_batch_size_box = tk.Spinbox(bottom_inputs_frame, from_=1, to=10000, font=('arial', 12, 'normal'), width=10, textvariable=tk.StringVar(self, value="10"))
        self.mini_batch_size_box.grid(row=1, column=1, sticky="w")

        # Learning rate input field
        tk.Label(bottom_inputs_frame, text='Learning rate:', font=('arial', 12, 'normal')).grid(row=2, column=0, sticky="e")
        self.learning_rate_box = tk.Entry(bottom_inputs_frame, font=('arial', 12, 'normal'))
        self.learning_rate_box.insert(0, "3")
        self.learning_rate_box.grid(row=2, column=1, sticky="w")

        # Regularization parameter input field
        tk.Label(bottom_inputs_frame, text='Regularization parameter:', font=('arial', 12, 'normal')).grid(row=3, column=0, sticky="e")
        self.regularization_box = tk.Entry(bottom_inputs_frame, font=('arial', 12, 'normal'))
        self.regularization_box.insert(0, "1")
        self.regularization_box.grid(row=3, column=1, sticky="w")

        # Number of epochs spinbox
        tk.Label(bottom_inputs_frame, text='Epochs:', font=('arial', 12, 'normal')).grid(row=4, column=0, sticky="e")
        self.epoch_box = tk.Spinbox(
            bottom_inputs_frame, from_=1, to=1000, font=('arial', 12, 'normal'), bg = '#FFFFFF', width=10, textvariable=tk.StringVar(self, value="5")
        )
        self.epoch_box.grid(row=4, column=1, sticky="w")

        # Stop after epoch checkbox
        self.stop_after_epoch_var = tk.IntVar()
        self.stop_after_epoch_box = tk.Checkbutton(bottom_inputs_frame, text='Stop after each epoch:', variable=self.stop_after_epoch_var, font=('arial', 12, 'normal'))
        self.stop_after_epoch_box.grid(row=5, column=0, columnspan=2, sticky="nsew", padx=50)

        # Number of tests after each epoch
        tk.Label(bottom_inputs_frame, text='Tests after each epoch', font=('arial', 12, 'normal')).grid(row=6, column=0, sticky="e")
        self.tests_num_box = tk.Spinbox(bottom_inputs_frame, from_=1, to=10000, font=('arial', 12, 'normal'), bg = '#FFFFFF', width=10, textvariable=tk.StringVar(self, value="10000"))
        self.tests_num_box.grid(row=6, column=1, sticky="w")

        ## Bottom buttons frame
        # 'Save' button
        self.save_button = tk.Button(bottom_buttons_frame, text='Save Network', font=('arial', 15, 'normal'), command=self.saveNetwork)
        self.save_button.grid(row=0, column=0, sticky="nsew")
        # 'Show graphs' button
        self.show_graphs_button =  tk.Button(bottom_buttons_frame, text='Show Progress', font=('arial', 15, 'normal'), command=lambda: self.controller.show_frame("ViewProgress"))
        self.show_graphs_button.grid(row=0, column=1, sticky="nsew")
        # 'Stop training' button
        self.resume_training_button = tk.Button(bottom_buttons_frame, text='No Network', font=('arial', 15, 'normal'), command=self.controller.continue_training)
        self.resume_training_button.grid(row=0, column=2, sticky="nsew")
        # 'Train' button
        self.train_button = tk.Button(bottom_buttons_frame, text='Start training', font=('arial', 15, 'normal'), command=self.trainNetwork)
        self.train_button.grid(row=0, column=3, sticky="nsew")
        # 'Browse outputs' button
        self.browse_outputs_button = tk.Button(bottom_buttons_frame, text='Browse Outputs', font=('arial', 15, 'normal'), command=self.browse_outputs)
        self.browse_outputs_button.grid(row=0, column=4, sticky="nsew")
        # 'Test own drawings' button
        self.draw_button = tk.Button(bottom_buttons_frame, text='Test own drawings', font=('arial', 15, 'normal'), command=lambda: self.controller.show_frame("CanvasDrawing"))
        self.draw_button.grid(row=0, column=5, sticky="nsew")

        self.update_elements()

    def activate_button(self, button, activation_val=True):
        if activation_val:
            state = "normal"
        else:
            state = "disabled"
        button["state"] = state

    def update_elements(self):
        network_needed_buttons = (self.save_button, self.draw_button, self.browse_outputs_button)

        for button in network_needed_buttons:
            ## Activate 'Save', 'Train', 'Browse outputs', 'Test own drawings' buttons if network is exists, but is not running
            self.activate_button(button, all((self.controller.network_created, not self.controller.training_running)))

        self.activate_button(self.create_network_button, not self.controller.training_running)
        self.activate_button(self.load_button, not self.controller.training_running)
        self.activate_button(self.show_graphs_button, self.controller.network_created)
        self.update_resume_button()
        self.update_train_button()

    def update_train_button(self):
        self.activate_button(self.train_button, self.controller.network_created)
        if self.controller.training_running or self.controller.epochs_to_run > 0:
            self.train_button["text"] = "Stop Training"
            self.train_button["command"] = self.stop_training
        else:
            self.train_button["text"] = "Train Network"
            self.train_button["command"] = self.trainNetwork

    def update_resume_button(self):
        # Activate 'Resume' button if network is not running but still has epochs in queue
        self.activate_button(self.resume_training_button, all((not self.controller.training_running, (self.controller.epochs_to_run > 0))))

        if self.controller.epochs_to_run > 0:
            self.resume_training_button["text"] = f"Continue ({self.controller.epochs_to_run} more epochs)"
        else:
            self.resume_training_button["text"] = f"Network idle"
        if self.controller.training_running:
            self.resume_training_button["text"] = f"Network running ({self.controller.epochs_to_run} more epochs)"
        if not self.controller.network_created:
            self.resume_training_button["text"] = f"No Network"

    def create_network(self):
        """After clicking the Create button"""
        input_sizes = list(map(int, self.sizes_input.get().replace(" ", "").split(",")))
        activation_func = self.activation_functions[self.activation_func_box.get()]
        cost_func = self.cost_functions[self.cost_func_box.get()]

        print(f"Creating a new neural network with sizes: {self.sizes_input.get()}")
        print(f"Activation function: {self.activation_func_box.get()}")
        print(f"Cost function: {self.cost_func_box.get()}")
        net = NeuralNetwork(sizes=input_sizes, cost_function=cost_func(), activation_function=activation_func)
        
        self.controller.update_network(net)

    def load_network(self):
        file_path = filedialog.askopenfilename(title="Load neural network",
            filetypes=[("JSON", ".json"), ("Text file", ".txt"), ("all files", ".*")]
        )
        if file_path != "":
            net = load_network(file_path)
            print(f"Loading a neural network from the file: " + file_path)
            print(f"Size: {str(tuple(net.sizes))}")
            self.controller.update_network(net)

    def trainNetwork(self):
        dataset_function = self.dataset_functions[self.dataset_box.get()]
        mini_batch_size = int(self.mini_batch_size_box.get())
        learning_rate = float(self.learning_rate_box.get())
        regularization = int(self.regularization_box.get())
        epochs = int(self.epoch_box.get())
        stop_after_epoch = self.stop_after_epoch_var.get()
        tests_after_epoch = int(self.tests_num_box.get())
        self.controller.initialize_training(dataset_function, mini_batch_size, learning_rate, regularization, epochs, stop_after_epoch, tests_after_epoch)

    def stop_training(self):
        update_elements_after_click = not self.controller.training_running  # If the button was clicked during pause -> update buttons immediatelly
        self.controller.stop_training()
        self.activate_button(self.train_button, False)
        if update_elements_after_click:
            self.update_elements()

    def saveNetwork(self):
        if self.controller.network is not None:
            file_name = filedialog.asksaveasfilename(title="Save neural network",
                filetypes=[("JSON", ".json"), ("Text file", ".txt"), ("all files", ".*")]
            )
            self.controller.network.save_network(file_name)
            print(f"Network saved to file {file_name}")

    def browse_outputs(self):
        print('clicked')


if __name__ == "__main__":
    from drawing_gui import NeuralNetworksGUI

    main_app = NeuralNetworksGUI()
    main_app.show_frame("MainScreen")
    main_app.mainloop()