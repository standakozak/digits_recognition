import tkinter as tk
import numpy as np
from PIL import Image, ImageTk


class BrowseOutputs(tk.Frame):
    WIDTH_SHOW_CANVAS = 560
    HEIGHT_SHOW_CANVAS = 560

    ARRAY_COLS = 28
    ARRAY_ROWS = 28

    def __init__(self, container, controller):
        tk.Frame.__init__(self, container)
        self.root = container
        self.controller = controller

        self.input_image = np.zeros((self.ARRAY_ROWS, self.ARRAY_COLS))

        self.canvas_show = tk.Canvas(self, bg="white", height=self.HEIGHT_SHOW_CANVAS, width=self.WIDTH_SHOW_CANVAS)
        self.canvas_show.grid(row=1, column=1, rowspan=2)

        tk.Button(self, text='Show next', bg='#FFFFFF', font=('arial', 15, 'normal'), command=self.controller.show_next_test_output).grid(row=3, column=0)
        tk.Button(self, text='Incorrect', bg='#FFFFFF', font=('arial', 15, 'normal'), command=lambda: self.controller.show_next_test_output(True)).grid(row=3, column=1)
        tk.Button(self, text='Main Screen', bg='#FFFFFF', font=('arial', 15, 'normal'), command=lambda: self.controller.show_frame("MainScreen")).grid(row=3, column=2)

        self.correct_answer_label = tk.Label(self, text="Correct answer: ", font=('arial', 15, 'normal'))
        self.correct_answer_label.grid(row=0, column=0, columnspan=2)
        self.output_label = tk.Label(self, text="", font=('arial', 15, 'normal'))
        self.output_label.grid(row=1, column=2, rowspan=2)

        self.controller.show_next_test_output()

    def show_output(self, image, desired_answer, real_probabilities, correct):
        self.input_image = image
        self.show_image()
        self.correct_answer_label["text"] = "Correct answer: " + desired_answer
        if not correct:
            self.correct_answer_label["fg"] = "red"
        else:
            self.correct_answer_label["fg"] = "black"
        self.display_probabilities(real_probabilities)

    def show_image(self):
        image_array = 255 * (1 - self.input_image)

        image = Image.fromarray(image_array.astype(np.uint8))
        self.resized_photo_image = ImageTk.PhotoImage(image.resize(size=(self.HEIGHT_SHOW_CANVAS, self.WIDTH_SHOW_CANVAS)))
        self.canvas_show.create_image(self.HEIGHT_SHOW_CANVAS//2, self.WIDTH_SHOW_CANVAS//2, image=self.resized_photo_image)

    def clear_frame(self, event):
        for widget in self.winfo_children():
            widget.destroy()

    def display_probabilities(self, probabilities):
        text = ""
        for answer, probability in probabilities:
            text += f"{str(answer)}: {round(float(probability), 2)} %\n"
        self.output_label["text"] = text

    def update_elements(self):
        self.controller.show_next_test_output()


if __name__ == "__main__":
    from drawing_gui import NeuralNetworksGUI

    main_app = NeuralNetworksGUI()
    main_app.show_frame("BrowseOutputs")
    main_app.mainloop()