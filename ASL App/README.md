# TensorFlow Lite Image Classification Android

<img src="../readme/android.jpg">

## Overview

In this android app, we have implemented a tensorflow lite model which can detect american sign language upto 29 signs. You can even select the model whether to run this on CPU or GPU.

## Model

In this app we have implemented the tensorflow model which is trained on MobileNetV2 architechture. It's inference time is around 175ms. The TF-Lite file is around 11MB

## Requirements

* Android Studio 3.2 (installed on a Linux, Mac or Windows machine)
* Android device in developer mode with USB debugging enabled
* USB cable (to connect Android device to your computer)

## Build and run

* Step 1. Clone the Repository
```
git clone https://github.com/sayannath/American-Sign-Language-Detection.git
```
* Step 2. Open Android Studio and from there choose 'Open an Existing Project'
```
The directory of the application is named as ASL App
```
* Step 3. Build the Android Studio project
```
Select Build -> Make Project and check that the project builds successfully. You will need Android SDK configured in the settings. You'll need at least SDK version 23. The build.gradle file will prompt you to download any missing libraries.
```
* Step 4. Install and Run the app
```
Connect the Android device to the computer and be sure to approve any ADB permission prompts that appear on your phone. Select Run -> Run app. Select the deployment target in the connected devices to the device on which the app will be installed. This will install the app on the device.
```
To test the app, open the app called ```American Sign Language App``` on your device. When you run the app the first time, the app will request permission to access the camera. Re-installing the app may require you to uninstall the previous installations.

## Branches
```
master - Implementated Tensorflow Lite
feature/tf-task-library - Implemented Tensorflow Lite Task Lite Library
```

## That's it!
