# Gym Exercise Detection for Adaptive Workout Music

**Bakalaura darbs:**  
*Trenažieru zāles vingrošanas noteikšana treniņam adaptīvai mūzikai*

**Bachelor's thesis:**  
*Gym Exercise Detection for Adaptive Workout Music*

This repository contains the code, datasets, analysis notebooks, and trained models used in the bachelor's thesis project.

The project focuses on detecting gym exercise and rest periods using smartwatch IMU sensor data. The detected exercise/rest signal is intended for use in adaptive workout music systems.

---

## Setup

Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html), then run:

```bash
conda env create --prefix ./envs -f environment.yml
conda activate ./envs
pip install -e .
```

---

## Related Applications

### WearOS smartwatch app

Real-time IMU data collection and exercise detection on a smartwatch.

<https://github.com/Markuss-B/GymActivityRecognition-WearOS>

### Phone annotator app

Phone application for managing recording sessions and annotating exercise start/end times.

<https://github.com/Markuss-B/GAR-Annotator>

### Initial ESP32 prototype

Initially developed as an ESP32/TensorFlow Lite Micro prototype with MPU6050 IMU data reading and BLE communication, later abandoned to develop the WearOS app for more practical real-time data collection and exercise recognition.

<https://github.com/Markuss-B/GAR_ESP_tflite_mpu6050_ble>

---

## Repository Structure

### `/src`

Contains code for preprocessing the RecoFit and MyoGym datasets and exporting them for model training in Kaggle.

Important file:

- `/src/data/pipeline.py` — prepares and exports datasets for training.

### `/data`

Contains the MyoGym, RecoFit, and collected thesis datasets.

### `/data/recordings`

Contains the collected workout recordings and analysis scripts.

Important files:

- `/data/recordings/analysis2.ipynb` final analysis notebook.
- `/data/recordings/analysis.ipynb` initial analysis experiments.
- `/data/recordings/recording.ipynb` sanity-checking session recordings.
- `/data/recordings/plot_recording.py` inspecting a processed session recording saved by `process_and_save_recording.py`.
- `/data/recordings/smoothing.py` and `/data/recordings/smoothing2.py` inspecting classification signal plots and post-processing behavior.

### `/models`

Contains trained models and scripts for converting models to TFLite format for smartwatch deployment.

---

## Kaggle Notebooks

### Training on the collected thesis dataset

<https://www.kaggle.com/code/markussbirznieks/gar-my-training>

### RecoFit + MyoGym training

<https://www.kaggle.com/code/markussbirznieks/gar-myogym-recofit-training>

### MyoGym training

<https://www.kaggle.com/code/markussbirznieks/gar-myogym-training>

### RecoFit training

<https://www.kaggle.com/code/markussbirznieks/gar-recofit-training>

### Offline testing of trained MyoGym/RecoFit models

<https://www.kaggle.com/code/markussbirznieks/gar-offline-model-test>
