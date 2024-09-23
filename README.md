# API Skin Disease Classifications

Welcome to the **Skin Disease Classifications API**! This API allows you to classify various skin diseases (Chickenpox, Cowpox, Healthy, HFMD, Measles, Monkeypox) using pre-trained **Hybrid CNN-SVM** models. This guide will help you get started with setting up and running the API on different devices, including Android emulators, iOS emulators, and physical devices.

> [!NOTE]
> The current API response is still using Indonesian language. English language response will be available in a future update.


## Install Dependencies.
Ensure that the following dependencies are installed in your environment:
- Flask (v3.0.3) – A lightweight web framework for building APIs.
- Tensorflow (v2.15.0) – A library for machine learning and deep learning.
- Joblib (v1.3.2) – A library for serializing Python objects (used for loading the trained model).

You can install these dependencies using `pip`:
```bash
pip install flask==3.0.3 tensorflow==2.15.0 joblib==1.3.2
```
> [!CAUTION]
> Make sure to use the same version to avoid compatibility issues and errors that may occur during execution.

## Run the API

#### If you use an Android or iOS emulator (e.g., Android Studio, Xcode):
```bash
flask run --host=0.0.0.0
```
#### If you use pyshical device:
1. Find your local IP address (e.g., `192.168.1.10`).
2. Run the API with the following command, replace the `x.x.x.x` with your local IP:
```bash
flask run --host=x.x.x.x
```
