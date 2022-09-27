import tkinter
from gui.CanvasDrawing import CanvasDrawing

if __name__ == "__main__":
    app = tkinter.Tk()
    frame = tkinter.Frame(app)
    frame.pack(side="top", expand=True, fill="both")
    CanvasDrawing(frame)
    app.mainloop()
