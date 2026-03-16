import tkinter as tk
import time
import csv
from datetime import datetime

class TrajectoryRecorder:
    def __init__(self, root):
        self.root = root
        self.root.title("Pen Trajectory Recorder")

        self.canvas = tk.Canvas(root, bg="white", width=600, height=600)
        self.canvas.pack()

        self.drawing = False
        self.trajectory = []
        self.start_time = None

        # Bind mouse events
        self.canvas.bind("<ButtonPress-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.stop_draw)

        # Bind keyboard shortcuts
        self.root.bind("s", self.save_trajectory)
        self.root.bind("c", self.clear_canvas)

        print("Instructions:")
        print("Draw with mouse.")
        print("Press 's' to save.")
        print("Press 'c' to clear.")

    def start_draw(self, event):
        self.drawing = True
        self.start_time = time.time()
        self.trajectory = []
        self.last_x, self.last_y = event.x, event.y

    def draw(self, event):
        if self.drawing:
            current_time = time.time() - self.start_time
            x, y = event.x, event.y

            # Store time + coordinates
            self.trajectory.append((current_time, x, y))

            # Draw line on canvas
            self.canvas.create_line(self.last_x, self.last_y, x, y, width=2)
            self.last_x, self.last_y = x, y

    def stop_draw(self, event):
        self.drawing = False

    def save_trajectory(self, event=None):
        if not self.trajectory:
            print("No trajectory to save.")
            return

        filename = f"trajectory_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["time_seconds", "x", "y"])
            writer.writerows(self.trajectory)

        print(f"Saved trajectory to {filename}")

    def clear_canvas(self, event=None):
        self.canvas.delete("all")
        self.trajectory = []
        print("Canvas cleared.")


if __name__ == "__main__":
    root = tk.Tk()
    app = TrajectoryRecorder(root)
    root.mainloop()