import tkinter as tk

class ViewProgress(tk.Frame):
    def __init__(self, container, controller):
        tk.Frame.__init__(self, container)
        self.controller = controller
        label = tk.Label(self, text="This is page 1")
        label.pack(side="top", fill="x", pady=10)
        button = tk.Button(self, text="Go to the start page",
                           command=lambda: controller.open_drawing())
        button.pack()