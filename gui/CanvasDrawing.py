import tkinter
import numpy as np
from PIL import Image, ImageDraw, ImageTk


class CanvasDrawing(tkinter.Frame):
    WIDTH_DRAW_CANVAS = 560
    HEIGHT_DRAW_CANVAS = 560

    WIDTH_SHOW_CANVAS = 280
    HEIGHT_SHOW_CANVAS = 280

    ARRAY_COLS = 28
    ARRAY_ROWS = 28

    WIDTH_FACTOR = WIDTH_DRAW_CANVAS // ARRAY_COLS
    HEIGHT_FACTOR = HEIGHT_DRAW_CANVAS // ARRAY_ROWS

    def __init__(self, container, controller):
        tkinter.Frame.__init__(self, container)
        self.root = container
        self.controller = controller

        self.canvas_draw = tkinter.Canvas(self, bg="white", height=self.HEIGHT_DRAW_CANVAS, width=self.WIDTH_DRAW_CANVAS)
        self.canvas_draw.grid(column="0", row="0")
        self.canvas_show = tkinter.Canvas(self, bg="white", height=self.HEIGHT_SHOW_CANVAS, width=self.WIDTH_SHOW_CANVAS)
        self.canvas_show.grid(column="1", row="0")

        self.pil_image = Image.new("P", (self.WIDTH_DRAW_CANVAS, self.HEIGHT_DRAW_CANVAS), (255, 255, 255))
        self.pil_draw = ImageDraw.Draw(self.pil_image)

        self.canvas_draw.bind("<Button-1>", self.get_x_and_y)
        self.canvas_draw.bind("<B1-Motion>", self.draw)
        self.canvas_draw.bind("<Button-3>", self.get_x_and_y)
        self.canvas_draw.bind("<B3-Motion>", self.clear)

        tkinter.Button(self, text='Back', bg='#FFFFFF', font=('arial', 15, 'normal'), command=lambda: self.controller.show_frame("MainScreen")).place(x=428, y=550)

    def create_resized_image(self):
        original_image = self.pil_image
        original_array = np.asarray(original_image)
        resized = []
        for row in range(0, self.HEIGHT_DRAW_CANVAS, self.HEIGHT_FACTOR):
            resized_row = []
            for col in range(0, self.WIDTH_DRAW_CANVAS, self.WIDTH_FACTOR):
                resized_row.append((original_array[row:row+self.HEIGHT_FACTOR, col:col+self.WIDTH_FACTOR]).mean())
            resized.append(resized_row)
        resized_arr = np.asarray(resized)
        resized_arr = 255 * (1 - resized_arr)

        resized_image = Image.fromarray(resized_arr.astype(np.uint8))
        self.resized_photo_image = ImageTk.PhotoImage(resized_image.resize(size=(self.HEIGHT_SHOW_CANVAS, self.WIDTH_SHOW_CANVAS)))

    def get_x_and_y(self, event):
        self.lasx, self.lasy = event.x, event.y

    def draw_on_both_canvases(self, event, width, color="black"):
        color_rgb = (0, 0, 0)
        if color == "white":
            color_rgb = (255, 255, 255)
        self.canvas_draw.create_line((self.lasx, self.lasy, event.x, event.y), 
                        fill=color, 
                        width=width)
        self.pil_draw.line((self.lasx, self.lasy, event.x, event.y), color_rgb, width=width)
        self.create_resized_image()
        self.canvas_show.create_image(self.HEIGHT_SHOW_CANVAS//2, self.WIDTH_SHOW_CANVAS//2, image=self.resized_photo_image)
        self.lasx, self.lasy = event.x, event.y

    def draw(self, event):
        self.draw_on_both_canvases(event=event, width=8, color="black")

    def clear(self, event):
        self.draw_on_both_canvases(event=event, width=16, color="white")
    
    def clear_frame(self, event):
        for widget in self.winfo_children():
            widget.destroy()

    def update_buttons(self, network_created=False, network_running=False, epochs_to_run=0):
        pass