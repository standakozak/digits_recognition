import tkinter as tk
import matplotlib

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

matplotlib.use("TkAgg")

class ViewProgress(tk.Frame):
    def __init__(self, container, controller):
        tk.Frame.__init__(self, container)
        self.controller = controller
        self.showing_accuracy = True
        self.grid_columnconfigure((0, 1, 2, 3, 4), weight=1)
        
        tk.Label(self, text='Training Progress', font=('arial', 24, 'normal')).grid(row=0, column=0, columnspan=5, sticky="nsew")

        plot_frame = tk.Frame(self)
        plot_frame.grid(row=1, column=0, rowspan=3, columnspan=5, sticky="nsew")

        figure = Figure()
        self.plot_canvas = FigureCanvasTkAgg(figure, plot_frame)
        self.axes = figure.add_subplot()

        self.plot_canvas.get_tk_widget().pack(fill="both")

        NavigationToolbar2Tk(self.plot_canvas, plot_frame).pack(fill="x", padx=100)

        self.switch_plot_button = tk.Button(self, text='Show cost', font=('arial', 15, 'normal'), command=self.switch_plot)
        self.switch_plot_button.grid(row=4, column=1, sticky="nsew")
        
        self.train_button = tk.Button(self, text='Main Screen', font=('arial', 15, 'normal'), command=lambda: self.controller.show_frame("MainScreen"))
        self.train_button.grid(row=4, column=2, sticky="nsew")

        # 'Stop training' button
        self.resume_training_button = tk.Button(self, text='No Network', font=('arial', 15, 'normal'), command=self.controller.continue_training)
        self.resume_training_button.grid(row=4, column=3, sticky="nsew")
        self.update_elements()

    def switch_plot(self):
        self.showing_accuracy = not self.showing_accuracy
        if self.showing_accuracy:
            self.switch_plot_button["text"] = "Show cost"
        else:
            self.switch_plot_button["text"] = "Show accuracy"
        self.update_elements()

    def update_elements(self):
        # Resume button activation
        self.activate_button(self.resume_training_button, all((not self.controller.training_running, (self.controller.epochs_to_run > 0))))
        if self.controller.epochs_to_run > 0:
            self.resume_training_button["text"] = f"Continue ({self.controller.epochs_to_run} more epochs)"
        else:
            self.resume_training_button["text"] = f"Network idle"
        if self.controller.training_running:
            self.resume_training_button["text"] = f"Network running ({self.controller.epochs_to_run} more epochs)"
        if not self.controller.network_created:
            self.resume_training_button["text"] = f"No Network"

        # Graph update
        if self.showing_accuracy:
            title = "Accuracy (%)"
            data = [[acc * 100 if acc is not None else acc for acc in self.controller.last_test_accuracies], 
                    [acc * 100 if acc is not None else acc for acc in self.controller.last_training_accuracies]
                    ]
        else:
            title = "Cost"
            data = [self.controller.last_test_costs, self.controller.last_training_costs]
        total_points = self.controller.total_training_epochs + 1
        data[0] = data[0] + [None for _ in range(total_points - len(data[0]))]
        data[1] = data[1] + [None for _ in range(total_points - len(data[1]))]
        x_axis = range(total_points)

        # Figure setup and plotting
        self.axes.clear()
        self.axes.set_title(f"Training and Test {title}")
        self.axes.set_xlabel("Epoch")
        self.axes.set_ylabel(title)

        self.axes.plot(x_axis, data[0], linestyle="-", marker="o", color="green", label="Test " + title.replace(" (%)", ""))
        self.axes.plot(x_axis, data[1], linestyle="-", marker="o", color="blue", label="Training " + title.replace(" (%)", ""))
        self.axes.grid(True)
        self.axes.legend()

        self.axes.set_xlim(left=0, right=total_points-1)

        self.plot_canvas.draw()

    def activate_button(self, button, activation_val=True):
        if activation_val:
            state = "normal"
        else:
            state = "disabled"
        button["state"] = state


if __name__ == "__main__":
    from drawing_gui import NeuralNetworksGUI

    main_app = NeuralNetworksGUI()
    #main_app.show_frame("ViewProgress")
    main_app.mainloop()