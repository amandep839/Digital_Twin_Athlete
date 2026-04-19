# 🏃‍♂️ Digital Twin Athlete 📌 Project Overview

Digital Twin Athlete is an AI-based smart athlete monitoring system that creates a virtual model (digital twin) of an athlete using simulated sensor inputs and machine learning predictions.

The system analyzes athlete performance data, predicts training zones, and visualizes results through an interactive dashboard and a connected Unity-based visualization environment.

This project integrates:

AI + IoT simulation + Streamlit dashboard + Unity visualization

to represent a real-world digital twin architecture.

---

## 🎯 Objectives

- Build a virtual athlete performance model
- Simulate sensor-based athlete data
- Predict athlete training zones using ML
- Display predictions using Streamlit dashboard
- Visualize athlete digital twin inside Unity application

---

## 🛠️ Technologies Used

- Python
- Machine Learning
- Streamlit Dashboard
- Pandas
- NumPy
- Scikit-learn
- Unity (3D visualization)
- Pickle (.pkl trained model files)
- CSV Dataset Processing

---

## 📂 Project Structure
Digital_Twin_Athlete/
│
├── simulator.py
├── inference.py
├── cleaned_dataset.csv
├── Model_Ready_Features_Fixed.csv
├── feature_cols.pkl
├── zone_classifier.pkl
│
├── Streamlit/
├── Experiment-01/
├── twin.html
└── README.md


---

## ⚙️ System Architecture

The project works in the following pipeline:

1. Athlete sensor data is simulated using `simulator.py`
2. Data is processed and passed to ML inference model
3. Model predicts athlete performance training zone
4. Results displayed on Streamlit dashboard
5. Data is sent to Unity application
6. Unity visualizes the athlete’s Digital Twin behavior

---

## 🔗 Unity Integration

The system is connected with a Unity-based Digital Twin visualization environment where:

- Athlete performance zones are reflected visually
- Real-time simulation updates athlete status
- Virtual athlete responds based on ML predictions
- Enables immersive monitoring experience

This creates a complete Digital Twin pipeline:

simulation → prediction → visualization

---

## 👨‍💻 Authors

Project Team – Digital Twin Athlete

- Amandeep Singh
- Rudresh Mande
- Viraj Jhadav
- Karan Jhadav
