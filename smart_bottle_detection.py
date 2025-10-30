import cv2
import numpy as np
from ultralytics import YOLO
import tkinter as tk
from PIL import Image, ImageTk
import sqlite3
import pyttsx3
from datetime import datetime
import threading
import logging
from tkinter import ttk

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class BottleDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Smart Plastic Bottle Detection and Reward System")
        self.root.geometry("1000x700")
        self.root.configure(bg="#f0f0f0")

        # Initialize YOLO model
        logging.info("Loading YOLOv8 model...")
        try:
            self.model = YOLO('yolov8n.pt')  # Replace with 'path/to/custom_plastic_bottle.pt' after training
        except Exception as e:
            logging.error(f"Failed to load YOLO model: {e}")
            raise
        self.bottle_class_id = 39  # Update to custom model class ID after training

        # Initialize webcam
        logging.info("Initializing webcam...")
        self.cap = None
        for index in [0, 1, 2]:
            self.cap = cv2.VideoCapture(index)
            if self.cap.isOpened():
                break
        if not self.cap or not self.cap.isOpened():
            raise Exception("Error: Could not open webcam at indices 0, 1, or 2.")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

        # Define Region of Interest (ROI) for the bin
        self.roi_x, self.roi_y = 80, 65  # Top center
        self.roi_width, self.roi_height = 160, 120

        # Initialize text-to-speech
        logging.info("Initializing text-to-speech...")
        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', 150)
            self.engine.setProperty('volume', 0.9)
        except Exception as e:
            logging.error(f"Text-to-speech initialization failed: {e}")

        # Initialize database
        logging.info("Setting up SQLite database...")
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

        # Initialize variables
        self.bottle_count = 0
        self.reward_per_bottle = 2  # ₹2 per bottle
        self.total_reward = 0
        self.detected_bottles = set()
        self.counted_bottles = set()
        self.running = True
        self.frame_skip = 2  # Process every 2nd frame for performance
        self.frame_count = 0

        # GUI elements
        self.main_frame = ttk.Frame(self.root, padding=10)
        self.main_frame.pack(fill="both", expand=True)

        self.video_label = ttk.Label(self.main_frame)
        self.video_label.pack(pady=10)

        self.count_label = ttk.Label(self.main_frame, text="Bottles Detected: 0", font=("Arial", 14))
        self.count_label.pack()

        self.reward_label = ttk.Label(self.main_frame, text="Total Reward: ₹0", font=("Arial", 14))
        self.reward_label.pack()

        self.alert_label = ttk.Label(self.main_frame, text="", font=("Arial", 12), foreground="red")
        self.alert_label.pack()

        self.reset_button = ttk.Button(self.main_frame, text="Reset Session", command=self.reset_session)
        self.reset_button.pack(pady=10)

        self.history_button = ttk.Button(self.main_frame, text="View Session History", command=self.show_history)
        self.history_button.pack(pady=5)

        # Start video processing
        logging.info("Starting video processing thread...")
        self.thread = threading.Thread(target=self.update_frame)
        self.thread.daemon = True
        self.thread.start()

    def update_frame(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                logging.error("Failed to capture frame from webcam.")
                break

            self.frame_count += 1
            if self.frame_count % self.frame_skip != 0:
                continue  # Skip frames for performance

            # Draw ROI (bin)
            cv2.rectangle(frame, (self.roi_x, self.roi_y),
                         (self.roi_x + self.roi_width, self.roi_y + self.roi_height),
                         (255, 0, 0), 2)
            cv2.putText(frame, 'Bin', (self.roi_x, self.roi_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            # Process frame with YOLO
            try:
                results = self.model(frame, conf=0.3)
                current_bottles = set()

                for result in results:
                    for box in result.boxes:
                        if int(box.cls) == self.bottle_class_id:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            conf = box.conf.item()

                            # Draw bounding box and label
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 250, 0), 2)
                            cv2.putText(frame, f'Bottle {conf:.2f}', (x1, y1-10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 250, 0), 2)

                            # Calculate centroid and area
                            centroid = (x1 + x2) // 2, (y1 + y2) // 2
                            area = (x2 - x1) * (y2 - y1)

                            # Check if bottle is in ROI
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
                        try:
                            self.engine.say("Warning: Bottle already counted. Please do not reuse bottles.")
                            self.engine.runAndWait()
                        except Exception as e:
                            logging.error(f"Text-to-speech error: {e}")
                        self.root.after(3000, lambda: self.alert_label.config(text=""))
                    else:
                        self.bottle_count += 1
                        self.total_reward = self.bottle_count * self.reward_per_bottle
                        self.counted_bottles.add(bottle)
                        try:
                            self.engine.say("Bottle detected")
                            self.engine.runAndWait()
                            self.engine.say("Thank you for recycling")
                            self.engine.runAndWait()
                        except Exception as e:
                            logging.error(f"Text-to-speech error: {e}")
                        self.count_label.config(text=f"Bottles Detected: {self.bottle_count}")
                        self.reward_label.config(text=f"Total Reward: ₹{self.total_reward}")

                self.detected_bottles = current_bottles

                # Convert frame to Tkinter format
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                img = img.resize((320, 240), Image.Resampling.LANCZOS)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)

                # Log session
                self.log_session()

            except Exception as e:
                logging.error(f"Error processing frame: {e}")

            self.root.update()
            cv2.waitKey(1)

    def log_session(self):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            self.cursor.execute('''
                INSERT INTO sessions (timestamp, bottle_count, reward)
                VALUES (?, ?, ?)
            ''', (timestamp, self.bottle_count, self.total_reward))
            self.conn.commit()
        except Exception as e:
            logging.error(f"Database error: {e}")

    def show_history(self):
        history_window = tk.Toplevel(self.root)
        history_window.title("Session History")
        history_window.geometry("600x400")

        tree = ttk.Treeview(history_window, columns=("ID", "Timestamp", "Bottles", "Reward"), show="headings")
        tree.heading("ID", text="Session ID")
        tree.heading("Timestamp", text="Timestamp")
        tree.heading("Bottles", text="Bottles Detected")
        tree.heading("Reward", text="Total Reward (₹)")
        tree.pack(fill="both", expand=True)

        scrollbar = ttk.Scrollbar(history_window, orient="vertical", command=tree.yview)
        scrollbar.pack(side="right", fill="y")
        tree.configure(yscrollcommand=scrollbar.set)

        try:
            self.cursor.execute("SELECT id, timestamp, bottle_count, reward FROM sessions")
            for row in self.cursor.fetchall():
                tree.insert("", "end", values=row)
        except Exception as e:
            logging.error(f"Error fetching session history: {e}")

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
            logging.error(f"Text-to-speech error: {e}")

    def cleanup(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.conn.close()
        self.root.quit()

if __name__ == "__main__":
    root = tk.Tk()
    app = BottleDetectionApp(root)
    root.protocol("WM_DELETE_WINDOW", app.cleanup)
    root.mainloop()