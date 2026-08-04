import tkinter as tk
import time
import csv
from datetime import datetime
from pathlib import Path
import sys

"""
Records the trajectory of handwriting digits.
Splits the trajectories into x and y trajectories
"""

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

OUTPUT_DIR = ROOT_DIR / "data/handwriting/digits_raw/"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def ask_for_digit(parent):
    """
    Modal popup asking 'Which digit have you drawn?'.
    Input is restricted to a single digit (0-9).
    Blocks until the user submits or closes the window.
    Returns the digit as an int, or None if cancelled/closed.
    """
    result = {"digit": None}

    dialog = tk.Toplevel(parent)
    dialog.title("Label Digit")
    dialog.geometry("300x140")
    dialog.resizable(False, False)
    dialog.transient(parent)
    dialog.grab_set()  # modal: blocks interaction with the main window

    def validate_digit(new_value):
        if new_value == "":
            return True
        return len(new_value) == 1 and new_value.isdigit()

    def submit():
        value = entry.get()
        if len(value) == 1 and value.isdigit():
            result["digit"] = int(value)
            dialog.destroy()
        else:
            error_label.config(text="Please enter exactly one digit (0-9).")

    def on_enter(event):
        submit()

    label = tk.Label(dialog, text="Which digit?", font=("Arial", 12))
    label.pack(pady=10)

    vcmd = (dialog.register(validate_digit), "%P")
    entry = tk.Entry(
        dialog,
        font=("Arial", 18),
        justify="center",
        width=3,
        validate="key",
        validatecommand=vcmd,
    )
    entry.pack(pady=5)
    entry.focus_set()
    entry.bind("<Return>", on_enter)

    error_label = tk.Label(dialog, text="", fg="red")
    error_label.pack()

    submit_btn = tk.Button(dialog, text="Submit", command=submit)
    submit_btn.pack(pady=5)

    parent.wait_window(dialog)  # blocks here until dialog closes
    return result["digit"]

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

        # Ask which digit was drawn before saving
        digit = ask_for_digit(self.root)
        if digit is None:
            print("Save cancelled: no digit label provided.")
            return

        filename = f"digit{digit}.csv"

        with open(OUTPUT_DIR / filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["digit_label", "time_seconds", "x", "y"])
            for t, x, y in self.trajectory:
                writer.writerow([digit, t, x, y])

        print(f"Saved trajectory (digit={digit}) to {filename}")

    def clear_canvas(self, event=None):
        self.canvas.delete("all")
        self.trajectory = []
        print("Canvas cleared.")


if __name__ == "__main__":
    root = tk.Tk()
    app = TrajectoryRecorder(root)
    root.mainloop()