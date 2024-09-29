import tkinter as tk
from tkinter import filedialog, messagebox
import subprocess
import threading

def browse_video():
    filename = filedialog.askopenfilename(title="Select Video File", filetypes=(("MP4 files", "*.mp4"), ("All files", "*.*")))
    video_path_entry.delete(0, tk.END)
    video_path_entry.insert(0, filename)

def run_command():
    video_path = video_path_entry.get()
    if not video_path:
        messagebox.showerror("Error", "Please select a video file.")
        return
    
    model_name = model_entry.get() or "yolov8n.pt"
    device = device_entry.get() or "cuda"
    confidence = confidence_entry.get() or "0.5"
    threshold = threshold_entry.get() or "150"
    
    command = f"python main.py --video \"{video_path}\" --model \"{model_name}\" --device \"{device}\" --confidence {confidence} --threshold {threshold}"
    
    def execute_command():
        try:
            subprocess.run(command, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            messagebox.showerror("Error", f"Command failed: {e}")
    
    # Start the command in a new thread to allow running multiple videos simultaneously
    thread = threading.Thread(target=execute_command)
    thread.start()

# Set up the GUI
root = tk.Tk()
root.title("YOLO Video Processing Interface")

tk.Label(root, text="Video Path:").grid(row=0, column=0, padx=10, pady=10)
video_path_entry = tk.Entry(root, width=50)
video_path_entry.grid(row=0, column=1, padx=10, pady=10)
tk.Button(root, text="Browse", command=browse_video).grid(row=0, column=2, padx=10, pady=10)

tk.Label(root, text="Model Name:").grid(row=1, column=0, padx=10, pady=10)
model_entry = tk.Entry(root)
model_entry.grid(row=1, column=1, padx=10, pady=10)
model_entry.insert(0, "yolov8n.pt")

tk.Label(root, text="Device:").grid(row=2, column=0, padx=10, pady=10)
device_entry = tk.Entry(root)
device_entry.grid(row=2, column=1, padx=10, pady=10)
device_entry.insert(0, "cuda")

tk.Label(root, text="Detection Confidence:").grid(row=3, column=0, padx=10, pady=10)
confidence_entry = tk.Entry(root)
confidence_entry.grid(row=3, column=1, padx=10, pady=10)
confidence_entry.insert(0, "0.5")

tk.Label(root, text="Threshold:").grid(row=4, column=0, padx=10, pady=10)
threshold_entry = tk.Entry(root)
threshold_entry.grid(row=4, column=1, padx=10, pady=10)
threshold_entry.insert(0, "100")

tk.Button(root, text="Run", command=run_command).grid(row=5, column=1, pady=20)

root.mainloop()
