# Set-point-temperature-based-HVAC-energy-consumption-prediction-model

- **HVAC energy consumption** constitutes a significant portion of total building energy usage.  
- In particular, changes in the set-point temperature have a **substantial impact on energy consumption**.  
- Understanding **how energy consumption varies with a 1°C increase or decrease in the set-point temperature** is **crucial for optimizing energy efficiency**.
# Measurement system structure
<p align="center">
  <img src="Image/Layers.png" width=700 alt="Layers">
</p>

# IoT measurement device 
This device consists of 3 parts, thermal environment measurement module, indoor air quality module(CO2, pm2.5,pm10) and text recognition module.
- For calculation of thermal comfort index you can check [**pmv**](https://github.com/Raskiller503/Thermal-comfort-tool-)
- For Image text recognition method by OpenCV you can check [**Set-point control panel**](https://github.com/Raskiller503/ImageRecognition-AC-pannel-_-OpenCV)
<p align="center">
  <img src="Image/device.png" width=300 alt="device">
</p>

## Complete Raspberry Pi Configuration Process

### 1. Download

Download the latest Raspbian installation system from the official website.

### 2. Format TF Card

Format the TF (microSD) card.

### 3. Burn System Image

Use **Win32DiskImager** to write the system image to the TF card.

### 4. Configure SSH & Wi-Fi

Create new `ssh` and `wpa_supplicant.conf` files to configure SSH and Wi-Fi.

### 5. Find Raspberry Pi IP Address

Use **IPScanner** to locate the Raspberry Pi's IP address.

### 6. Connect via PuTTY

Use **PuTTY** to connect to the Raspberry Pi. Ensure you are on the same Wi-Fi network but connect using different IP addresses.

- The default password for the first login is `raspberrypi`.

### 7. Run `sudo raspi-config`

Configure your Raspberry Pi with the following settings:

1. **Interface Options** – Enable interfaces like SSH, Camera, SPI, GPIO, etc.
2. **Expand Filesystem** – Expand the filesystem to use the entire TF card.
3. **GPU Memory** – Set GPU memory to 256MB.
4. **Localization Options** – Set locale to `ZH-UTF-8` and timezone to `Asia/Tokyo`.
5. **Network Options** – Configure the network name.

### 8. Install XRDP

Run the following commands to install XRDP:

```bash
sudo apt update
sudo apt install xrdp 
```
## Dataset
<div style="display: flex; justify-content: space-between align=center;">
    <img src="Image/Hourly_MRT.png" alt="Hourly_MRT" width="47%">
    <img src="Image/Hourly_PMV.png" alt="Hourly_PMV" width="47%">
</div>

## Correlation
The dataset was collected in Summer, we could find that set-point temperature has the most negative influence to HVAC energy consumption.
<p align="center">
  <img src="Image/Heatmap.png" width=700  alt="Correlation">
</p>

# Machine Learning model algorithm

This project implements a machine learning pipeline to predict HVAC (Heating, Ventilation, and Air Conditioning) energy consumption based on indoor and outdoor environmental parameters. It leverages various machine learning models, such as Logistic Regression, Gradient Boosting Classifier, and Support Vector Machine (SVM), to analyze data and make predictions.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)<!-- - [Usage](#usage)- [Visualization](#visualization) -->
- [Model Performance](#model-performance)
- [Future Work](#future-work)

---

## Overview

This project uses real-world HVAC data collected from sensors. The primary objective is to develop a reliable prediction model for energy consumption, providing actionable insights for energy-saving and operational efficiency in HVAC systems.

---

## Features

- Support for multiple machine learning models, including:
  - Logistic Regression
  - Gradient Boosting Classifier
  - Support Vector Machine (SVM)
- Hyperparameter tuning with cross-validation.
- Visualization of predictions, training data, and decision boundaries.
- Compatibility with multi-dimensional feature sets.
- 3D visualization for selected feature combinations.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Raskiller503/Set-point-temperature-based-AC-energy-consumption-prediction-model/blob/main/Code/Cross-Validation.py
    ```
## Model Performance

<p align="center">
  <img src="Image/四种模型.png" width=500  alt="result">
</p>

## Future Work
We have proposed one Model Predicitive Control(MPC) model based on thermal comfort & HVAC energy, which will be open access later.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
