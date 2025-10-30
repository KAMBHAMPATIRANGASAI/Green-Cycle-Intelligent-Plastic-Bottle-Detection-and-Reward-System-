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
import os
import time
import queue
import json
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bottle_detection.log'),
        logging.StreamHandler()
    ]
)

class BottleDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Smart Plastic Bottle Detection and Reward System v3.0")
        self.root.geometry("1400x900")
        self.root.configure(bg="#f0f0f0")
        
        # Initialize variables
        self.model = None
        self.cap = None
        self.engine = None
        self.conn = None
        self.cursor = None
        self.bottle_count = 0
        self.reward_per_bottle = 2  # ₹2 per bottle
        self.total_reward = 0
        self.donor_name = ""  # Store donor name
        self.detected_bottles = set()
        self.counted_bottles = set()
        self.running = False
        self.frame_skip = 2
        self.frame_count = 0
        self.last_detection_time = {}
        self.detection_cooldown = 2.0
        self.current_session_id = None
        
        # Frame and ROI settings
        self.frame_width, self.frame_height = 640, 480
        self.roi_width, self.roi_height = 200, 150
        self.roi_x = (self.frame_width - self.roi_width) // 2
        self.roi_y = (self.frame_height - self.roi_height) // 2
        self.bottle_class_id = 39
        
        # Threading and queue
        self.frame_queue = queue.Queue(maxsize=2)
        self.gui_queue = queue.Queue()
        self.tts_queue = queue.Queue()
        
        # Configuration
        self.config = self.load_config()
        
        # Initialize components
        self.setup_logging()
        self.setup_database()
        self.setup_model()
        self.setup_webcam()
        self.setup_tts()
        self.setup_gui()
        self.setup_threads()
        
        # Start processing
        self.start_detection()

    def load_config(self):
        """Load configuration from file or create default"""
        config_file = Path("bottle_detection_config.json")
        default_config = {
            "reward_per_bottle": 2,
            "detection_confidence": 0.4,
            "roi_width": 200,
            "roi_height": 150,
            "detection_cooldown": 2.0,
            "frame_skip": 2,
            "tts_enabled": True,
            "auto_save_interval": 30
        }
        
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                    for key, value in default_config.items():
                        if key not in config:
                            config[key] = value
                    return config
            except Exception as e:
                logging.error(f"Error loading config: {e}")
        
        with open(config_file, 'w') as f:
            json.dump(default_config, f, indent=2)
        
        return default_config

    def save_config(self):
        """Save current configuration"""
        try:
            with open("bottle_detection_config.json", 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            logging.error(f"Error saving config: {e}")

    def setup_logging(self):
        """Enhanced logging setup"""
        self.logger = logging.getLogger(__name__)
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / f"bottle_detection_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(file_handler)
        self.logger.info("Application started at 10:54 AM IST, July 12, 2025")

    def setup_model(self):
        """Enhanced model setup with error handling"""
        self.logger.info("Loading YOLOv8 model...")
        try:
            model_paths = ['custom_plastic_bottle.pt', 'yolov8n.pt', 'yolov8s.pt']
            model_loaded = False
            
            for model_path in model_paths:
                if os.path.exists(model_path):
                    try:
                        self.model = YOLO(model_path)
                        self.bottle_class_id = 0 if 'custom_plastic_bottle' in model_path else 39
                        self.logger.info(f"Successfully loaded model: {model_path}")
                        model_loaded = True
                        break
                    except Exception as e:
                        self.logger.warning(f"Failed to load {model_path}: {e}")
                        continue
            
            if not model_loaded:
                self.logger.info("Downloading YOLOv8n model...")
                self.model = YOLO('yolov8n.pt')
                self.bottle_class_id = 39
                
        except Exception as e:
            self.logger.error(f"Failed to load any YOLO model: {e}")
            messagebox.showerror("Error", f"Failed to load YOLO model: {e}")
            raise

    def setup_webcam(self):
        """Enhanced webcam setup with multiple camera support"""
        self.logger.info("Initializing webcam...")
        for index in range(5):
            try:
                self.cap = cv2.VideoCapture(index)
                if self.cap.isOpened():
                    ret, _ = self.cap.read()
                    if ret:
                        self.logger.info(f"Successfully opened camera at index {index}")
                        break
                    else:
                        self.cap.release()
            except Exception as e:
                self.logger.warning(f"Failed to open camera at index {index}: {e}")
                continue
        
        if not self.cap or not self.cap.isOpened():
            self.logger.error("Could not open any webcam.")
            messagebox.showerror("Error", "Could not open webcam. Please check your camera connection.")
            raise RuntimeError("No webcam available")
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def setup_tts(self):
        """Enhanced TTS setup with error handling"""
        self.logger.info("Initializing text-to-speech...")
        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', 150)
            self.engine.setProperty('volume', 0.8)
            voices = self.engine.getProperty('voices')
            if voices:
                for voice in voices:
                    if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                        self.engine.setProperty('voice', voice.id)
                        break
                else:
                    self.engine.setProperty('voice', voices[0].id)
            self.tts_enabled = self.config.get('tts_enabled', True)
        except Exception as e:
            self.logger.error(f"Text-to-speech initialization failed: {e}")
            self.engine = None
            self.tts_enabled = False

    def setup_database(self):
        """Set up SQLite database with donor name"""
        self.logger.info("Setting up SQLite database...")
        try:
            db_dir = Path("data")
            db_dir.mkdir(exist_ok=True)
            self.conn = sqlite3.connect('data/bottle_detection_new.db', check_same_thread=False)
            self.cursor = self.conn.cursor()
            
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    donor_name TEXT,
                    start_time TEXT NOT NULL,
                    end_time TEXT,
                    total_bottles INTEGER DEFAULT 0,
                    total_reward REAL DEFAULT 0
                )
            ''')
            
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER,
                    detection_time TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    x_center INTEGER,
                    y_center INTEGER,
                    area INTEGER,
                    FOREIGN KEY (session_id) REFERENCES sessions (id) ON DELETE CASCADE
                )
            ''')
            
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            ''')
            
            self.conn.commit()
            self.logger.info("Database setup completed successfully")
            
            # Initialize current session ID
            self.current_session_id = self.get_current_session_id()
            if self.current_session_id is None:
                self.current_session_id = 1  # Start with ID 1 if no sessions exist
            
        except Exception as e:
            self.logger.error(f"Database setup error: {e}")
            messagebox.showerror("Error", f"Failed to set up database: {e}")
            raise

    def setup_gui(self):
        """Enhanced GUI setup with donor name input"""
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('Title.TLabel', font=('Arial', 16, 'bold'), background='#f0f0f0')
        style.configure('Stats.TLabel', font=('Arial', 12, 'bold'), background='#f0f0f0')
        style.configure('Alert.TLabel', font=('Arial', 11, 'bold'), foreground='red', background='#f0f0f0')
        style.configure('Success.TLabel', font=('Arial', 11, 'bold'), foreground='green', background='#f0f0f0')
        
        main_container = ttk.Frame(self.root, padding=10)
        main_container.pack(fill="both", expand=True)
        
        title_label = ttk.Label(main_container, text="Smart Plastic Bottle Detection & Reward System", 
                               style='Title.TLabel')
        title_label.pack(pady=(0, 10))
        
        donor_frame = ttk.Frame(main_container)
        donor_frame.pack(fill="x", pady=5)
        ttk.Label(donor_frame, text="Donor Name:", style='Stats.TLabel').pack(side="left")
        self.donor_entry = ttk.Entry(donor_frame)
        self.donor_entry.pack(side="left", padx=5)
        ttk.Button(donor_frame, text="Set Name", command=self.set_donor_name).pack(side="left")
        
        main_paned = ttk.PanedWindow(main_container, orient='horizontal')
        main_paned.pack(fill="both", expand=True)
        
        left_frame = ttk.Frame(main_paned)
        main_paned.add(left_frame, weight=2)
        
        video_frame = ttk.LabelFrame(left_frame, text="Live Camera Feed", padding=10)
        video_frame.pack(fill="both", expand=True, pady=(0, 10))
        
        self.video_label = ttk.Label(video_frame, text="Initializing camera...", 
                                    font=('Arial', 12), anchor='center')
        self.video_label.pack(expand=True)
        
        camera_controls = ttk.Frame(left_frame)
        camera_controls.pack(fill="x", pady=5)
        
        ttk.Button(camera_controls, text="📷 Capture", command=self.capture_frame).pack(side="left", padx=5)
        ttk.Button(camera_controls, text="🎥 Toggle Camera", command=self.toggle_camera).pack(side="left", padx=5)
        
        right_frame = ttk.Frame(main_paned)
        main_paned.add(right_frame, weight=1)
        
        stats_frame = ttk.LabelFrame(right_frame, text="Session Statistics", padding=10)
        stats_frame.pack(fill="x", pady=(0, 10))
        
        self.donor_label = ttk.Label(stats_frame, text="Donor: Not set", style='Stats.TLabel')
        self.donor_label.pack(anchor="w", pady=2)
        
        self.count_label = ttk.Label(stats_frame, text="Bottles Detected: 0", style='Stats.TLabel')
        self.count_label.pack(anchor="w", pady=2)
        
        self.reward_label = ttk.Label(stats_frame, text="Total Reward: ₹0", style='Stats.TLabel')
        self.reward_label.pack(anchor="w", pady=2)
        
        self.session_time_label = ttk.Label(stats_frame, text="Session Time: 00:00:00", style='Stats.TLabel')
        self.session_time_label.pack(anchor="w", pady=2)
        
        ttk.Label(stats_frame, text="Detection Confidence:", style='Stats.TLabel').pack(anchor="w", pady=(10, 2))
        self.confidence_var = tk.DoubleVar()
        self.confidence_progress = ttk.Progressbar(stats_frame, variable=self.confidence_var, 
                                                  maximum=100, length=200)
        self.confidence_progress.pack(anchor="w", pady=2)
        
        self.alert_label = ttk.Label(stats_frame, text="", style='Alert.TLabel')
        self.alert_label.pack(anchor="w", pady=(10, 0))
        
        controls_frame = ttk.LabelFrame(right_frame, text="Controls", padding=10)
        controls_frame.pack(fill="x", pady=(0, 10))
        
        ttk.Button(controls_frame, text="🔄 Reset Session", 
                  command=self.reset_session).pack(fill="x", pady=2)
        ttk.Button(controls_frame, text="📊 View History", 
                  command=self.show_history).pack(fill="x", pady=2)
        ttk.Button(controls_frame, text="⚙️ Settings", 
                  command=self.show_settings).pack(fill="x", pady=2)
        
        roi_frame = ttk.LabelFrame(right_frame, text="Detection Zone", padding=10)
        roi_frame.pack(fill="x", pady=(0, 10))
        
        ttk.Label(roi_frame, text="Adjust detection zone:").pack(anchor="w")
        roi_controls = ttk.Frame(roi_frame)
        roi_controls.pack(fill="x", pady=5)
        
        ttk.Button(roi_controls, text="↑", command=lambda: self.adjust_roi(0, -10)).pack(side="top")
        ttk.Button(roi_controls, text="←", command=lambda: self.adjust_roi(-10, 0)).pack(side="left")
        ttk.Button(roi_controls, text="→", command=lambda: self.adjust_roi(10, 0)).pack(side="right")
        ttk.Button(roi_controls, text="↓", command=lambda: self.adjust_roi(0, 10)).pack(side="bottom")
        
        self.session_start_time = time.time()

    def set_donor_name(self):
        """Set the donor name for the session"""
        name = self.donor_entry.get().strip()
        if not name:
            messagebox.showwarning("Invalid Input", "Please enter a valid donor name.")
            return
        self.donor_name = name
        self.donor_label.config(text=f"Donor: {self.donor_name}")
        self.logger.info(f"Donor name set to: {self.donor_name}")
        self.show_alert(f"Donor name set to: {self.donor_name}", "success")

    def setup_threads(self):
        """Setup worker threads"""
        self.logger.info("Setting up worker threads...")
        self.tts_thread = threading.Thread(target=self.tts_worker, daemon=True)
        self.tts_thread.start()
        self.gui_thread = threading.Thread(target=self.gui_worker, daemon=True)
        self.gui_thread.start()

    def tts_worker(self):
        """Worker thread for TTS operations"""
        while True:
            try:
                message = self.tts_queue.get(timeout=1)
                if message == "STOP":
                    break
                if self.engine and self.tts_enabled:
                    self.engine.say(message)
                    self.engine.runAndWait()
                self.tts_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"TTS error: {e}")

    def gui_worker(self):
        """Worker thread for GUI updates"""
        while True:
            try:
                update_func = self.gui_queue.get(timeout=1)
                if update_func == "STOP":
                    break
                self.root.after_idle(update_func)
                self.gui_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"GUI update error: {e}")

    def start_detection(self):
        """Start the detection process"""
        if not self.running:
            self.running = True
            self.detection_thread = threading.Thread(target=self.detection_loop, daemon=True)
            self.detection_thread.start()
            self.logger.info("Detection started")

    def stop_detection(self):
        """Stop the detection process"""
        self.running = False
        self.logger.info("Detection stopped")

    def detection_loop(self):
        """Main detection loop"""
        while self.running:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    self.logger.warning("Failed to capture frame")
                    time.sleep(0.1)
                    continue
                
                self.frame_count += 1
                if self.frame_count % self.frame_skip != 0:
                    continue
                
                processed_frame = self.process_frame(frame.copy())
                self.gui_queue.put(lambda: self.update_video_display(processed_frame))
                self.gui_queue.put(self.update_session_time)
                
            except Exception as e:
                self.logger.error(f"Error in detection loop: {e}")
                time.sleep(0.1)

    def process_frame(self, frame):
        """Process a single frame for bottle detection"""
        try:
            cv2.rectangle(frame, (self.roi_x, self.roi_y),
                         (self.roi_x + self.roi_width, self.roi_y + self.roi_height),
                         (0, 255, 0), 3)
            cv2.putText(frame, 'Detection Zone', (self.roi_x, self.roi_y - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            results = self.model(frame, conf=self.config.get('detection_confidence', 0.4), verbose=False)
            
            current_bottles = set()
            max_confidence = 0
            
            for result in results:
                if result.boxes is None:
                    continue
                    
                for box in result.boxes:
                    class_id = int(box.cls.item())
                    if class_id == self.bottle_class_id:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        conf = box.conf.item()
                        max_confidence = max(max_confidence, conf)
                        
                        color = (0, 255, 0) if conf > 0.6 else (0, 165, 255)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, f'Bottle {conf:.2f}', (x1, y1-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        
                        centroid = ((x1 + x2) // 2, (y1 + y2) // 2)
                        area = (x2 - x1) * (y2 - y1)
                        
                        if (self.roi_x <= centroid[0] <= self.roi_x + self.roi_width and
                            self.roi_y <= centroid[1] <= self.roi_y + self.roi_height):
                            
                            bottle_signature = (centroid[0], centroid[1], area, conf)
                            current_bottles.add(bottle_signature)
                            cv2.circle(frame, centroid, 5, (255, 0, 0), -1)
            
            self.gui_queue.put(lambda: self.confidence_var.set(max_confidence * 100))
            self.process_detections(current_bottles)
            
            cv2.putText(frame, f'Bottles: {self.bottle_count} | Reward: ₹{self.total_reward}', 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            return frame
            
        except Exception as e:
            self.logger.error(f"Error processing frame: {e}")
            return frame

    def process_detections(self, current_bottles):
        """Process detected bottles and update counts"""
        current_time = time.time()
        
        for bottle in current_bottles:
            cx, cy, area, conf = bottle
            is_new_bottle = True
            bottle_key = f"{cx}_{cy}_{area}"
            
            for counted_bottle in list(self.counted_bottles):
                cx_c, cy_c, area_c, _ = counted_bottle
                distance = np.sqrt((cx - cx_c)**2 + (cy - cy_c)**2)
                area_diff = abs(area - area_c)
                
                if distance < 50 and area_diff < 2000:
                    is_new_bottle = False
                    break
            
            if bottle_key in self.last_detection_time:
                time_diff = current_time - self.last_detection_time[bottle_key]
                if time_diff < self.detection_cooldown:
                    is_new_bottle = False
            
            if is_new_bottle and conf > 0.5:
                self.count_new_bottle(bottle, current_time, bottle_key)

    def count_new_bottle(self, bottle, current_time, bottle_key):
        """Count a new bottle detection"""
        cx, cy, area, conf = bottle
        
        self.bottle_count += 1
        self.total_reward = self.bottle_count * self.reward_per_bottle
        self.counted_bottles.add(bottle)
        self.last_detection_time[bottle_key] = current_time
        
        self.gui_queue.put(lambda: self.count_label.config(text=f"Bottles Detected: {self.bottle_count}"))
        self.gui_queue.put(lambda: self.reward_label.config(text=f"Total Reward: ₹{self.total_reward}"))
        
        success_msg = f"Bottle #{self.bottle_count} detected! +₹{self.reward_per_bottle}"
        self.gui_queue.put(lambda: self.show_alert(success_msg, "success"))
        
        tts_msg = f"Bottle {self.bottle_count} detected. Reward earned: {self.reward_per_bottle} rupees. Thank you for recycling!"
        self.tts_queue.put(tts_msg)
        
        self.log_detection(cx, cy, area, conf)
        self.logger.info(f"New bottle detected: #{self.bottle_count}, Confidence: {conf:.2f}")

    def show_alert(self, message, alert_type="info"):
        """Show alert message with auto-clear"""
        if alert_type == "success":
            self.alert_label.config(text=message, style='Success.TLabel')
        else:
            self.alert_label.config(text=message, style='Alert.TLabel')
        self.root.after(3000, lambda: self.alert_label.config(text=""))

    def update_video_display(self, frame):
        """Update video display in GUI"""
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img = img.resize((640, 480), Image.Resampling.LANCZOS)
            imgtk = ImageTk.PhotoImage(image=img)
            
            self.video_label.configure(image=imgtk, text="")
            self.video_label.image = imgtk
            
        except Exception as e:
            self.logger.error(f"Error updating video display: {e}")

    def update_session_time(self):
        """Update session timer"""
        try:
            elapsed = time.time() - self.session_start_time
            hours = int(elapsed // 3600)
            minutes = int((elapsed % 3600) // 60)
            seconds = int(elapsed % 60)
            time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            self.session_time_label.config(text=f"Session Time: {time_str}")
        except Exception as e:
            self.logger.error(f"Error updating session time: {e}")

    def adjust_roi(self, dx, dy):
        """Adjust ROI position"""
        new_x = max(0, min(self.frame_width - self.roi_width, self.roi_x + dx))
        new_y = max(0, min(self.frame_height - self.roi_height, self.roi_y + dy))
        self.roi_x = new_x
        self.roi_y = new_y
        self.logger.info(f"ROI adjusted to: ({self.roi_x}, {self.roi_y})")

    def capture_frame(self):
        """Capture and save current frame"""
        try:
            ret, frame = self.cap.read()
            if ret:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"captures/bottle_detection_{timestamp}.jpg"
                os.makedirs("captures", exist_ok=True)
                cv2.imwrite(filename, frame)
                self.show_alert(f"Frame saved: {filename}", "success")
                self.logger.info(f"Frame captured: {filename}")
        except Exception as e:
            self.logger.error(f"Error capturing frame: {e}")
            self.show_alert("Failed to capture frame", "error")

    def toggle_camera(self):
        """Toggle camera on/off"""
        if self.running:
            self.stop_detection()
            self.show_alert("Camera stopped", "info")
        else:
            self.start_detection()
            self.show_alert("Camera started", "success")

    def log_detection(self, cx, cy, area, conf):
        """Log detection to database"""
        try:
            detection_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.cursor.execute('''
                INSERT INTO detections (session_id, detection_time, confidence, x_center, y_center, area)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (self.current_session_id, detection_time, conf, cx, cy, area))
            self.conn.commit()
        except Exception as e:
            self.logger.error(f"Database logging error: {e}")

    def log_session(self):
        """Log session to database with donor name"""
        try:
            end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            start_time = datetime.fromtimestamp(self.session_start_time).strftime("%Y-%m-%d %H:%M:%S")
            
            # Fallback to "Unknown" if donor_name is not set
            if not self.donor_name:
                self.donor_name = "Unknown"
            
            self.cursor.execute('''
                INSERT INTO sessions (donor_name, start_time, end_time, total_bottles, total_reward)
                VALUES (?, ?, ?, ?, ?)
            ''', (self.donor_name, start_time, end_time, self.bottle_count, self.total_reward))
            self.conn.commit()
            self.current_session_id = self.cursor.lastrowid
            self.logger.info(f"Session logged: Donor={self.donor_name}, Bottles={self.bottle_count}, Reward=₹{self.total_reward}, ID={self.current_session_id}")
        except Exception as e:
            self.logger.error(f"Session logging error: {e}")

    def get_current_session_id(self):
        """Get or create the current session ID"""
        self.cursor.execute("SELECT id FROM sessions ORDER BY id DESC LIMIT 1")
        result = self.cursor.fetchone()
        return result[0] if result else None

    def delete_session(self, tree, history_window):
        """Delete selected session from database and reindex IDs"""
        try:
            selected_item = tree.selection()
            if not selected_item:
                messagebox.showwarning("No Selection", "Please select a session to delete.", parent=history_window)
                return
            
            if messagebox.askyesno("Confirm Delete", "Are you sure you want to delete the selected session?", parent=history_window):
                session_id = tree.item(selected_item)['values'][0]
                self.cursor.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
                self.conn.commit()
                self.logger.info(f"Deleted session ID: {session_id}")
                
                self.cursor.execute("SELECT id FROM sessions ORDER BY id")
                remaining_ids = [row[0] for row in self.cursor.fetchall()]
                
                if remaining_ids:
                    self.cursor.execute("DELETE FROM sqlite_sequence WHERE name='sessions'")
                    for new_id, old_id in enumerate(remaining_ids, start=1):
                        self.cursor.execute("""
                            UPDATE sessions SET id = ? WHERE id = ?
                        """, (new_id, old_id))
                        self.cursor.execute("""
                            UPDATE detections SET session_id = ? WHERE session_id = ?
                        """, (new_id, old_id))
                    self.conn.commit()
                
                self.refresh_history_window(history_window)
                messagebox.showinfo("Success", "Session deleted successfully.", parent=history_window)
                
        except Exception as e:
            self.logger.error(f"Error deleting session: {e}")
            messagebox.showerror("Error", f"Failed to delete session: {e}", parent=history_window)

    def show_history(self):
        """Show session history window with donor name"""
        try:
            history_window = tk.Toplevel(self.root)
            history_window.title("Session History")
            history_window.geometry("1000x700")
            history_window.configure(bg="#f0f0f0")
            
            notebook = ttk.Notebook(history_window)
            notebook.pack(fill="both", expand=True, padx=10, pady=10)
            
            sessions_frame = ttk.Frame(notebook)
            notebook.add(sessions_frame, text="Sessions")
            
            tree_frame = ttk.Frame(sessions_frame)
            tree_frame.pack(fill="both", expand=True, pady=10)
            
            columns = ("ID", "Donor Name", "Start Time", "End Time", "Bottles", "Reward")
            tree = ttk.Treeview(tree_frame, columns=columns, show="headings", height=15)
            
            tree.heading("ID", text="Session ID")
            tree.heading("Donor Name", text="Donor Name")
            tree.heading("Start Time", text="Start Time")
            tree.heading("End Time", text="End Time")
            tree.heading("Bottles", text="Bottles Detected")
            tree.heading("Reward", text="Total Reward (₹)")
            
            tree.column("ID", width=80, anchor="center")
            tree.column("Donor Name", width=150, anchor="center")
            tree.column("Start Time", width=150, anchor="center")
            tree.column("End Time", width=150, anchor="center")
            tree.column("Bottles", width=100, anchor="center")
            tree.column("Reward", width=100, anchor="center")
            
            tree.pack(side="left", fill="both", expand=True)
            
            v_scrollbar = ttk.Scrollbar(tree_frame, orient="vertical", command=tree.yview)
            v_scrollbar.pack(side="right", fill="y")
            tree.configure(yscrollcommand=v_scrollbar.set)
            
            h_scrollbar = ttk.Scrollbar(sessions_frame, orient="horizontal", command=tree.xview)
            h_scrollbar.pack(fill="x")
            tree.configure(xscrollcommand=h_scrollbar.set)
            
            try:
                self.cursor.execute("SELECT id, donor_name, start_time, end_time, total_bottles, total_reward FROM sessions ORDER BY start_time DESC")
                for row in self.cursor.fetchall():
                    tree.insert("", "end", values=row)
                    
            except Exception as e:
                self.logger.error(f"Error fetching session history: {e}")
                messagebox.showerror("Error", "Failed to load session history", parent=history_window)
            
            stats_frame = ttk.Frame(notebook)
            notebook.add(stats_frame, text="Statistics")
            
            stats_container = ttk.Frame(stats_frame, padding=20)
            stats_container.pack(fill="both", expand=True)
            
            try:
                self.cursor.execute("""
                    SELECT 
                        COUNT(*) as total_sessions,
                        SUM(total_bottles) as total_bottles,
                        SUM(total_reward) as total_rewards,
                        AVG(total_bottles) as avg_bottles_per_session,
                        MAX(total_bottles) as max_bottles_session
                    FROM sessions
                """)
                
                stats = self.cursor.fetchone()
                if stats and stats[0] > 0:
                    total_sessions, total_bottles, total_rewards, avg_bottles, max_bottles = stats
                    stats_text = f"""
                    📊 SYSTEM STATISTICS
                    
                    Total Sessions: {total_sessions or 0}
                    Total Bottles Detected: {total_bottles or 0}
                    Total Rewards Earned: ₹{total_rewards or 0:.2f}
                    Average Bottles per Session: {avg_bottles or 0:.1f}
                    Best Session (Most Bottles): {max_bottles or 0}
                    """
                    stats_label = ttk.Label(stats_container, text=stats_text, 
                                          font=('Arial', 11), justify='left')
                    stats_label.pack(anchor="w", pady=10)
                else:
                    no_data_label = ttk.Label(stats_container, 
                                            text="No session data available yet.", 
                                            font=('Arial', 12))
                    no_data_label.pack(pady=50)
                    
            except Exception as e:
                self.logger.error(f"Error calculating statistics: {e}")
            
            charts_frame = ttk.Frame(notebook)
            notebook.add(charts_frame, text="Charts")
            
            try:
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
                fig.suptitle('Bottle Detection Analytics', fontsize=16, fontweight='bold')
                
                # Chart 1: Bottles per Session
                self.cursor.execute("SELECT id, total_bottles FROM sessions ORDER BY start_time")
                session_data = self.cursor.fetchall()
                if session_data:
                    session_ids, bottles = zip(*session_data)
                    ax1.bar(range(len(session_ids)), bottles, color='#FF6B6B')
                    ax1.set_title('Bottles per Session')
                    ax1.set_xlabel('Session ID')
                    ax1.set_ylabel('Bottles Detected')
                    ax1.grid(True, alpha=0.3)
                
                # Chart 2: Reward per Session
                if session_data:
                    rewards = [row[1] * self.reward_per_bottle for row in session_data]
                    ax2.bar(range(len(session_ids)), rewards, color='#4ECDC4')
                    ax2.set_title('Reward per Session')
                    ax2.set_xlabel('Session ID')
                    ax2.set_ylabel('Total Reward (₹)')
                    ax2.grid(True, alpha=0.3)
                
                # Chart 4: Daily Bottle Detection (All Time) - Moved to ax4 position
                self.cursor.execute("""
                    SELECT DATE(start_time) as date, SUM(total_bottles) as daily_bottles
                    FROM sessions 
                    GROUP BY DATE(start_time)
                    ORDER BY date
                """)
                daily_data = self.cursor.fetchall()
                if daily_data:
                    dates, bottles = zip(*daily_data)
                    ax4.plot(range(len(dates)), bottles, marker='o', color='#45B7D1', linewidth=2, markersize=6)
                    ax4.set_title('Daily Bottle Detection (All Time)')
                    ax4.set_xlabel('Days')
                    ax4.set_ylabel('Bottles Detected')
                    ax4.grid(True, alpha=0.3)
                
                # Chart 3: Average Bottles per Session Over Time - Moved to ax3 position
                self.cursor.execute("SELECT start_time, total_bottles FROM sessions ORDER BY start_time")
                session_data = self.cursor.fetchall()
                if session_data:
                    dates = [datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S") for row in session_data]
                    bottles = [row[1] for row in session_data]
                    cumulative_avg = [sum(bottles[:i+1]) / (i+1) for i in range(len(bottles))]
                    ax3.plot(dates, cumulative_avg, color='#F7B32B', linewidth=2)
                    ax3.set_title('Average Bottles per Session (Cumulative)')
                    ax3.set_xlabel('Date')
                    ax3.set_ylabel('Average Bottles')
                    ax3.grid(True, alpha=0.3)
                    plt.xticks(rotation=45)
                
                plt.tight_layout()
                canvas = FigureCanvasTkAgg(fig, master=charts_frame)
                canvas.draw()
                canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)
                
            except Exception as e:
                self.logger.error(f"Error creating charts: {e}")
                error_label = ttk.Label(charts_frame, 
                                      text="Unable to generate charts. Please check the data.", 
                                      font=('Arial', 12))
                error_label.pack(pady=50)
            
            buttons_frame = ttk.Frame(sessions_frame)
            buttons_frame.pack(fill="x", pady=10)
            
            ttk.Button(buttons_frame, text="🔄 Refresh Data", 
                      command=lambda: self.refresh_history_window(history_window)).pack(side="left", padx=5)
            ttk.Button(buttons_frame, text="🗑️ Delete Session", 
                      command=lambda: self.delete_session(tree, history_window)).pack(side="left", padx=5)
        
        except Exception as e:
            self.logger.error(f"Error showing history: {e}")
            messagebox.showerror("Error", "Failed to open history window")

    def refresh_history_window(self, window):
        """Refresh history window data"""
        window.destroy()
        self.show_history()

    def show_settings(self):
        """Show settings configuration window"""
        settings_window = tk.Toplevel(self.root)
        settings_window.title("Settings")
        settings_window.geometry("500x600")
        settings_window.configure(bg="#f0f0f0")
        
        main_container = ttk.Frame(settings_window, padding=20)
        main_container.pack(fill="both", expand=True)
        
        ttk.Label(main_container, text="Detection Settings", 
                 font=('Arial', 14, 'bold')).pack(pady=(0, 15))
        
        conf_frame = ttk.Frame(main_container)
        conf_frame.pack(fill="x", pady=5)
        ttk.Label(conf_frame, text="Detection Confidence:").pack(side="left")
        
        self.conf_var = tk.DoubleVar(value=self.config.get('detection_confidence', 0.4))
        conf_scale = ttk.Scale(conf_frame, from_=0.1, to=0.9, 
                              variable=self.conf_var, orient="horizontal")
        conf_scale.pack(side="right", fill="x", expand=True, padx=(10, 0))
        
        self.conf_label = ttk.Label(conf_frame, text=f"{self.conf_var.get():.2f}")
        self.conf_label.pack(side="right", padx=(5, 10))
        
        def update_conf_label(*args):
            self.conf_label.config(text=f"{self.conf_var.get():.2f}")
        self.conf_var.trace('w', update_conf_label)
        
        reward_frame = ttk.Frame(main_container)
        reward_frame.pack(fill="x", pady=5)
        ttk.Label(reward_frame, text="Reward per Bottle (₹):").pack(side="left")
        
        self.reward_var = tk.DoubleVar(value=self.config.get('reward_per_bottle', 2))
        reward_spin = ttk.Spinbox(reward_frame, from_=0.5, to=10.0, increment=0.5,
                                 textvariable=self.reward_var, width=10)
        reward_spin.pack(side="right")
        
        cooldown_frame = ttk.Frame(main_container)
        cooldown_frame.pack(fill="x", pady=5)
        ttk.Label(cooldown_frame, text="Detection Cooldown (seconds):").pack(side="left")
        
        self.cooldown_var = tk.DoubleVar(value=self.config.get('detection_cooldown', 2.0))
        cooldown_spin = ttk.Spinbox(cooldown_frame, from_=1.0, to=10.0, increment=0.5,
                                   textvariable=self.cooldown_var, width=10)
        cooldown_spin.pack(side="right")
        
        ttk.Label(main_container, text="ROI Size", 
                 font=('Arial', 12, 'bold')).pack(pady=(20, 10))
        
        roi_size_frame = ttk.Frame(main_container)
        roi_size_frame.pack(fill="x", pady=5)
        
        ttk.Label(roi_size_frame, text="Width:").pack(side="left")
        self.roi_width_var = tk.IntVar(value=self.roi_width)
        width_spin = ttk.Spinbox(roi_size_frame, from_=100, to=400, increment=10,
                                textvariable=self.roi_width_var, width=10)
        width_spin.pack(side="left", padx=(5, 20))
        
        ttk.Label(roi_size_frame, text="Height:").pack(side="left")
        self.roi_height_var = tk.IntVar(value=self.roi_height)
        height_spin = ttk.Spinbox(roi_size_frame, from_=100, to=300, increment=10,
                                 textvariable=self.roi_height_var, width=10)
        height_spin.pack(side="left", padx=(5, 0))
        
        ttk.Label(main_container, text="Audio Settings", 
                 font=('Arial', 12, 'bold')).pack(pady=(20, 10))
        
        self.tts_enabled_var = tk.BooleanVar(value=self.config.get('tts_enabled', True))
        tts_check = ttk.Checkbutton(main_container, text="Enable Text-to-Speech", 
                                   variable=self.tts_enabled_var)
        tts_check.pack(anchor="w", pady=5)
        
        ttk.Label(main_container, text="Performance Settings", 
                 font=('Arial', 12, 'bold')).pack(pady=(20, 10))
        
        frame_skip_frame = ttk.Frame(main_container)
        frame_skip_frame.pack(fill="x", pady=5)
        ttk.Label(frame_skip_frame, text="Frame Skip (higher = better performance):").pack(side="left")
        
        self.frame_skip_var = tk.IntVar(value=self.config.get('frame_skip', 2))
        skip_spin = ttk.Spinbox(frame_skip_frame, from_=1, to=10, increment=1,
                               textvariable=self.frame_skip_var, width=10)
        skip_spin.pack(side="right")
        
        button_frame = ttk.Frame(main_container)
        button_frame.pack(fill="x", pady=(30, 0))
        
        ttk.Button(button_frame, text="Apply Settings", 
                  command=lambda: self.apply_settings(settings_window)).pack(side="right", padx=(5, 0))
        ttk.Button(button_frame, text="Cancel", 
                  command=settings_window.destroy).pack(side="right")
        ttk.Button(button_frame, text="Reset to Defaults", 
                  command=self.reset_settings).pack(side="left")

    def apply_settings(self, window):
        """Apply settings and close window"""
        try:
            self.config['detection_confidence'] = self.conf_var.get()
            self.config['reward_per_bottle'] = self.reward_var.get()
            self.config['detection_cooldown'] = self.cooldown_var.get()
            self.config['roi_width'] = self.roi_width_var.get()
            self.config['roi_height'] = self.roi_height_var.get()
            self.config['tts_enabled'] = self.tts_enabled_var.get()
            self.config['frame_skip'] = self.frame_skip_var.get()
            
            self.reward_per_bottle = self.config['reward_per_bottle']
            self.detection_cooldown = self.config['detection_cooldown']
            self.roi_width = self.config['roi_width']
            self.roi_height = self.config['roi_height']
            self.tts_enabled = self.config['tts_enabled']
            self.frame_skip = self.config['frame_skip']
            
            self.total_reward = self.bottle_count * self.reward_per_bottle
            self.reward_label.config(text=f"Total Reward: ₹{self.total_reward}")
            
            self.save_config()
            window.destroy()
            self.show_alert("Settings applied successfully!", "success")
            self.logger.info("Settings updated")
            
        except Exception as e:
            self.logger.error(f"Error applying settings: {e}")
            messagebox.showerror("Error", f"Failed to apply settings: {e}")

    def reset_settings(self):
        """Reset settings to defaults"""
        if messagebox.askyesno("Confirm Reset", "Reset all settings to defaults?"):
            self.conf_var.set(0.4)
            self.reward_var.set(2.0)
            self.cooldown_var.set(2.0)
            self.roi_width_var.set(200)
            self.roi_height_var.set(150)
            self.tts_enabled_var.set(True)
            self.frame_skip_var.set(2)

    def reset_session(self):
        """Reset current session"""
        if messagebox.askyesno("Confirm Reset", "Reset current session? This will clear all counts."):
            self.log_session()  # Ensure session is saved before reset
            
            self.bottle_count = 0
            self.total_reward = 0
            self.donor_name = ""
            self.detected_bottles.clear()
            self.counted_bottles.clear()
            self.last_detection_time.clear()
            self.current_session_id = self.get_current_session_id()
            if self.current_session_id is None:
                self.current_session_id = 1  # Start new session if no previous sessions
            
            self.count_label.config(text="Bottles Detected: 0")
            self.reward_label.config(text="Total Reward: ₹0")
            self.donor_label.config(text="Donor: Not set")
            self.alert_label.config(text="")
            self.confidence_var.set(0)
            self.donor_entry.delete(0, tk.END)
            
            self.session_start_time = time.time()
            
            self.tts_queue.put("Session reset. Ready for new detections.")
            self.show_alert("Session reset successfully!", "success")
            self.logger.info("Session reset")

    def cleanup(self):
        """Cleanup resources before closing"""
        self.logger.info("Cleaning up resources...")
        
        try:
            self.running = False
            self.tts_queue.put("STOP")
            self.gui_queue.put("STOP")
            
            if self.bottle_count > 0:
                self.log_session()  # Save final session if any bottles detected
            
            if self.cap:
                self.cap.release()
            
            if self.conn:
                self.conn.close()
            
            if self.engine:
                self.engine.stop()
            
            self.logger.info("Cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
        
        finally:
            self.root.quit()

def main():
    """Main function to run the application"""
    try:
        root = tk.Tk()
        app = BottleDetectionApp(root)
        root.protocol("WM_DELETE_WINDOW", app.cleanup)
        root.mainloop()
        
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        messagebox.showerror("Fatal Error", f"Application failed to start: {e}")

if __name__ == "__main__":
    main()