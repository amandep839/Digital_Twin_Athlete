🏃‍♂️ Digital Twin Athlete
📌 Project Overview

Digital Twin Athlete is an AI-based smart athlete monitoring system that creates a virtual model (digital twin) of an athlete using simulated sensor inputs and machine learning predictions. The system analyzes athlete performance data, predicts training zones, and visualizes results through an interactive dashboard and a connected Unity-based visualization environment.

This project integrates AI + IoT simulation + Streamlit dashboard + Unity visualization to represent a real-world digital twin architecture.

🎯 Objectives
Build a virtual athlete performance model
Simulate sensor-based athlete data
Predict athlete training zones using ML
Display predictions using Streamlit dashboard
Visualize athlete digital twin inside Unity application
🛠️ Technologies Used
Python
Machine Learning
Streamlit Dashboard
Pandas
NumPy
Scikit-learn
Unity (3D visualization)
Pickle (.pkl trained model files)
CSV Dataset Processing
📂 Project Structure
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
│
├── Experiment-01/
│
├── twin.html
│
└── README.md
⚙️ System Architecture

The project works in the following pipeline:

1️⃣ Athlete sensor data is simulated using simulator.py
2️⃣ Data is processed and passed to ML inference model
3️⃣ Model predicts athlete performance training zone
4️⃣ Results displayed on Streamlit dashboard
5️⃣ Data is sent to Unity application
6️⃣ Unity visualizes the athlete’s Digital Twin behavior

🔗 Unity Integration

The system is connected with a Unity-based Digital Twin visualization environment where:

Athlete performance zones are reflected visually
Real-time simulation updates athlete status
Virtual athlete responds based on ML predictions
Enables immersive monitoring experience

This creates a complete Digital Twin pipeline from simulation → prediction → visualization

🚀 How to Run the Project
Step 1: Clone Repository
git clone https://github.com/your-username/Digital_Twin_Athlete.git
Step 2: Install Dependencies
pip install pandas numpy scikit-learn streamlit
Step 3: Run Simulator
python simulator.py
Step 4: Run Prediction Script
python inference.py
Step 5: Launch Streamlit Dashboard
streamlit run Streamlit/app.py
Step 6: Run Unity Visualization

Open the Unity project and start the scene to view the Digital Twin athlete responding to prediction outputs.

📊 Features

✅ Athlete sensor simulation
✅ Machine learning zone classification
✅ Feature engineering pipeline
✅ Streamlit visualization dashboard
✅ Unity-based Digital Twin visualization
✅ End-to-end AI + IoT Digital Twin workflow

📈 Applications
Smart sports analytics
Athlete training optimization
Injury prevention monitoring
Fitness intelligence systems
Digital twin simulation research
🔮 Future Improvements
Real-time wearable sensor integration
Cloud-based deployment
Mobile monitoring dashboard
Deep learning prediction upgrade
Live Unity–Python socket communication pipeline

Project Team – Digital Twin Athlete

Amandeep Singh
Rudresh Mande
Viraj Jhadav
Karan Jhadav

AI + IoT + Machine Learning + Unity Integration Project 🚀


Digital Twin Athlete – AI + IoT + Unity Integration Project 🚀
