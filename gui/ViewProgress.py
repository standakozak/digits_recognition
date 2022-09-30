import tkinter as tk

class ViewProgress(tk.Frame):
    def __init__(self, container, controller):
        tk.Frame.__init__(self, container)
        self.controller = controller
        label = tk.Label(self, text="Showing graphs")
        label.pack(side="top", fill="x", pady=10)
        button = tk.Button(self, text="Main Screen",
                           command=lambda: controller.show_frame("MainScreen"))
        button.pack()
        self.update_elements()

    def update_elements(self):
        print("Last data:")
        print(f"Last training costs: {self.controller.last_training_costs}")
        print(f"Last test costs: {self.controller.last_test_costs}")
        print(f"Last training accuracies: {self.controller.last_training_accuracies}")
        print(f"Last test accuracies: {self.controller.last_test_accuracies}")