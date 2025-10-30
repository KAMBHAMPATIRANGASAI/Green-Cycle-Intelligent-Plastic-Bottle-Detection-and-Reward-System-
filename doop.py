import cv2
import numpy as np
from ultralytics import YOLO
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import sqlite3
import pyttsx3
from datetime import datetime
import threading
import logging
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class BottleDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Smart Plastic Bottle Detection and Reward System")
        self.root.geometry("1200x800")
        self.root.configure(bg="#e6e6e6")

        # Initialize variables
        self.model = None
        self.cap = None
        self.engine = None
        self.conn = None
        self.cursor = None
        self.bottle_count = 0
        self.reward_per_bottle = 2  # ₹2 per bottle
        self.total_reward = 0
        self.detected_bottles = set()
        self.counted_bottles = set()
        self.running = True
        self.frame_skip = 3
        self.frame_count = 0
        # Center ROI in 640x480 frame
        frame_width, frame_height = 640, 480
        self.roi_width, self.roi_height = 160, 120
        self.roi_x = (frame_width - self.roi_width) // 2  # Center horizontally
        self.roi_y = (frame_height - self.roi_height) // 2  # Center vertically
        self.bottle_class_id = 39  # Update after custom model training

        # Initialize components
        self.setup_logging()
        self.setup_database()
        self.setup_model()
        self.setup_webcam()
        self.setup_tts()
        self.setup_gui()
        self.setup_thread()

    def setup_logging(self):
        self.logger = logging.getLogger()
        file_handler = logging.FileHandler('bottle_detection.log')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(file_handler)

    def setup_model(self):
        self.logger.info("Loading YOLOv8 model...")
        try:
            model_path = 'custom_plastic_bottle.pt' if os.path.exists('custom_plastic_bottle.pt') else 'yolov8n.pt'
            self.model = YOLO(model_path)
            self.bottle_class_id = 0 if 'custom_plastic_bottle.pt' in model_path else 39
        except Exception as e:
            self.logger.error(f"Failed to load YOLO model: {e}")
            messagebox.showerror("Error", "Failed to load YOLO model. Please check the model file.")
            raise

    def setup_webcam(self):
        self.logger.info("Initializing webcam...")
        for index in range(3):
            self.cap = cv2.VideoCapture(index)
            if self.cap.isOpened():
                break
        if not self.cap or not self.cap.isOpened():
            self.logger.error("Could not open webcam.")
            messagebox.showerror("Error", "Could not open webcam. Please check your camera.")
            raise
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)  # Set frame rate for smoother display

    def setup_tts(self):
        self.logger.info("Initializing text-to-speech...")
        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', 150)
            self.engine.setProperty('volume', 0.9)
            voices = self.engine.getProperty('voices')
            self.engine.setProperty('voice', voices[0].id)
        except Exception as e:
            self.logger.error(f"Text-to-speech initialization failed: {e}")

    def setup_database(self):
        self.logger.info("Setting up SQLite database...")
        try:
            self.conn = sqlite3.connect('bottle_detection.db')
            self.cursor = self.conn.cursor()
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    bottle_count INTEGER,
                    reward REAL
                )
            ''')
            self.conn.commit()
        except Exception as e:
            self.logger.error(f"Database setup error: {e}")
            messagebox.showerror("Error", "Failed to set up database.")
            raise

    def setup_gui(self):
        self.main_frame = ttk.Frame(self.root, padding=10, style='Main.TFrame')
        self.main_frame.pack(fill="both", expand=True)

        # Style configuration
        style = ttk.Style()
        style.configure('Main.TFrame', background='#e6e6e6')
        style.configure('TButton', font=('Arial', 12), padding=10)
        style.configure('TLabel', font=('Arial', 14), background='#e6e6e6')

        # Video display
        self.video_label = ttk.Label(self.main_frame)
        self.video_label.pack(pady=10)

        # Stats frame
        self.stats_frame = ttk.Frame(self.main_frame)
        self.stats_frame.pack(fill="x", pady=10)
        self.count_label = ttk.Label(self.stats_frame, text="Bottles Detected: 0")
        self.count_label.pack(side="left", padx=20)
        self.reward_label = ttk.Label(self.stats_frame, text="Total Reward: ₹0")
        self.reward_label.pack(side="left", padx=20)

        # Alert label
        self.alert_label = ttk.Label(self.main_frame, text="", foreground="red")
        self.alert_label.pack(pady=5)

        # Buttons frame
        self.buttons_frame = ttk.Frame(self.main_frame)
        self.buttons_frame.pack(pady=10)
        ttk.Button(self.buttons_frame, text="Reset Session", command=self.reset_session).pack(side="left", padx=5)
        ttk.Button(self.buttons_frame, text="View History", command=self.show_history).pack(side="left", padx=5)
        ttk.Button(self.buttons_frame, text="Export History", command=self.export_history).pack(side="left", padx=5)

    def setup_thread(self):
        self.logger.info("Starting video processing thread...")
        self.thread = threading.Thread(target=self.update_frame)
        self.thread.daemon = True
        self.thread.start()

    def update_frame(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                self.logger.error("Failed to capture frame from webcam.")
                break

            self.frame_count += 1
            if self.frame_count % self.frame_skip != 0:
                continue

            # Draw ROI (centered bin)
            cv2.rectangle(frame, (self.roi_x, self.roi_y),
                         (self.roi_x + self.roi_width, self.roi_y + self.roi_height),
                         (255, 0, 0), 2)
            cv2.putText(frame, 'Bin', (self.roi_x, self.roi_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            try:
                results = self.model(frame, conf=0.4)
                current_bottles = set()

                for result in results:
                    for box in result.boxes:
                        if int(box.cls) == self.bottle_class_id:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            conf = box.conf.item()
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(frame, f'Bottle {conf:.2f}', (x1, y1-10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                            centroid = (x1 + x2) // 2, (y1 + y2) // 2
                            area = (x2 - x1) * (y2 - y1)
                            if (self.roi_x <= centroid[0] <= self.roi_x + self.roi_width and
                                self.roi_y <= centroid[1] <= self.roi_y + self.roi_height):
                                bottle_signature = (centroid[0], centroid[1], area)
                                current_bottles.add(bottle_signature)

                # Update bottle count
                new_bottles = current_bottles - self.detected_bottles
                for bottle in new_bottles:
                    cx, cy, ca = bottle
                    is_reused = False
                    for counted_bottle in self.counted_bottles:
                        cx_c, cy_c, ca_c = counted_bottle
                        if (abs(cx - cx_c) < 20 and abs(cy - cy_c) < 20 and abs(ca - ca_c) < 1000):
                            is_reused = True
                            break
                    if is_reused:
                        self.alert_label.config(text="Warning: Bottle already counted!")
                        self.engine.say("Warning: Bottle already counted.")
                        self.engine.runAndWait()
                        self.root.after(3000, lambda: self.alert_label.config(text=""))
                    else:
                        self.bottle_count += 1
                        self.total_reward = self.bottle_count * self.reward_per_bottle
                        self.counted_bottles.add(bottle)
                        self.engine.say("Bottle detected. Thank you for recycling.")
                        self.engine.runAndWait()
                        self.count_label.config(text=f"Bottles Detected: {self.bottle_count}")
                        self.reward_label.config(text=f"Total Reward: ₹{self.total_reward}")
                        self.log_session()

                self.detected_bottles = current_bottles

                # Display frame with synchronization
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                img = img.resize((640, 480), Image.Resampling.LANCZOS)
                imgtk = ImageTk.PhotoImage(image=img)
                self.root.after_idle(self.update_video_label, imgtk)

            except Exception as e:
                self.logger.error(f"Error processing frame: {e}")

            cv2.waitKey(1)

    def update_video_label(self, imgtk):
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

    def log_session(self):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            self.cursor.execute('''
                INSERT INTO sessions (timestamp, bottle_count, reward)
                VALUES (?, ?, ?)
            ''', (timestamp, self.bottle_count, self.total_reward))
            self.conn.commit()
        except Exception as e:
            self.logger.error(f"Database error: {e}")

    def show_history(self):
        history_window = tk.Toplevel(self.root)
        history_window.title("Session History")
        history_window.geometry("800x600")

        # Treeview for session data
        tree_frame = ttk.Frame(history_window)
        tree_frame.pack(fill="both", expand=True, pady=10)
        tree = ttk.Treeview(tree_frame, columns=("ID", "Timestamp", "Bottles", "Reward"), show="headings")
        tree.heading("ID", text="Session ID")
        tree.heading("Timestamp", text="Timestamp")
        tree.heading("Bottles", text="Bottles Detected")
        tree.heading("Reward", text="Total Reward (₹)")
        tree.pack(side="left", fill="both", expand=True)

        scrollbar = ttk.Scrollbar(tree_frame, orient="vertical", command=tree.yview)
        scrollbar.pack(side="right", fill="y")
        tree.configure(yscrollcommand=scrollbar.set)

        # Fetch data
        try:
            self.cursor.execute("SELECT id, timestamp, bottle_count, reward FROM sessions")
            for row in self.cursor.fetchall():
                tree.insert("", "end", values=row)
        except Exception as e:
            self.logger.error(f"Error fetching session history: {e}")

        # Plot trend
        fig, ax = plt.subplots(figsize=(6, 4))
        try:
            self.cursor.execute("SELECT timestamp, bottle_count FROM sessions ORDER BY timestamp")
            data = self.cursor.fetchall()
            if data:
                df = pd.DataFrame(data, columns=['Timestamp', 'Bottles'])
                df['Timestamp'] = pd.to_datetime(df['Timestamp'])
                ax.plot(df['Timestamp'], df['Bottles'], marker='o')
                ax.set_title("Bottle Detection Trend")
                ax.set_xlabel("Timestamp")
                ax.set_ylabel("Bottles Detected")
                plt.xticks(rotation=45)
                canvas = FigureCanvasTkAgg(fig, master=history_window)
                canvas.draw()
                canvas.get_tk_widget().pack(pady=10)
        except Exception as e:
            self.logger.error(f"Error plotting trend: {e}")

    def export_history(self):
        try:
            self.cursor.execute("SELECT id, timestamp, bottle_count, reward FROM sessions")
            df = pd.DataFrame(self.cursor.fetchall(), columns=['Session ID', 'Timestamp', 'Bottles', 'Reward'])
            filename = f"session_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(filename, index=False)
            messagebox.showinfo("Success", f"History exported to {filename}")
        except Exception as e:
            self.logger.error(f"Error exporting history: {e}")
            messagebox.showerror("Error", "Failed to export history.")

    def reset_session(self):
        self.bottle_count = 0
        self.total_reward = 0
        self.detected_bottles.clear()
        self.counted_bottles.clear()
        self.count_label.config(text="Bottles Detected: 0")
        self.reward_label.config(text="Total Reward: ₹0")
        self.alert_label.config(text="")
        try:
            self.engine.say("Session reset")
            self.engine.runAndWait()
        except Exception as e:
            self.logger.error(f"Text-to-speech error: {e}")

    def cleanup(self):
        self.running = False
        if self.cap:
            self.cap.release()
        if self.conn:
            self.conn.close()
        self.root.quit()

if __name__ == "__main__":
    root = tk.Tk()
    app = BottleDetectionApp(root)
    root.protocol("WM_DELETE_WINDOW", app.cleanup)
    root.mainloop()