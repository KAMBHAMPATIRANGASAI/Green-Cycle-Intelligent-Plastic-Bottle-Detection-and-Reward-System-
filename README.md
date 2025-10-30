# 🌿 Green-Cycle: Intelligent Plastic Bottle Detection and Reward System

### 🎓 MCA Data Science "Computer Vision" Project
**Author:** K. Ranga Sai  
**Guide:** Dr. V. Harsha Shastri, Associate Professor, Department of MCA, Aurora University  

---

## 📘 Overview
Green-Cycle is an AI-based system that detects plastic bottles in real-time using YOLOv8 and rewards users ₹2 per bottle to promote recycling.  
It includes a Tkinter GUI, SQLite database for session tracking, and voice feedback using Text-to-Speech.

---

## 🧠 Features
- Real-time plastic bottle detection using YOLOv8  
- Tkinter GUI with live video feed  
- ₹2 reward per detected bottle  
- SQLite database for saving donor and detection data  
- Text-to-speech (pyttsx3) for audio feedback  
- Analytics with Matplotlib (charts and trends)  
- Multi-threaded performance for smooth running  

---

## ⚙️ Requirements
### Hardware
- Webcam (minimum 720p)
- Intel i5 or above
- 8 GB RAM

### Software
- Windows 10/11
- Python 3.8+
- Libraries: OpenCV, ultralytics, Tkinter, pyttsx3, SQLite3, NumPy, Matplotlib, Pillow

---

## 🧬 How It Works

Captures video from webcam

Detects plastic bottles using YOLOv8 (confidence ≥ 0.4)

Displays detections in GUI

Calculates reward (₹2 per bottle)

Logs data into SQLite database

Provides voice feedback and charts

## 📊 Results

Detection Accuracy: ~86%

Smooth real-time detection

User-friendly interface with live stats

Database logging and trend visualization

## 🌍 Community Impact

Promotes recycling through rewards

Encourages eco-friendly behavior

Provides real data for waste management analysis

## ⚠️ Limitations

Works best in good lighting

Only supports one webcam

Detects only plastic bottles
