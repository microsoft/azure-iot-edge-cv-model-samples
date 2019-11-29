# MLOps Implementation for YOLOv3

## Prerequisites
* An [Azure account](https://account.microsoft.com/account?lang=en-us)
* An Azure subscription
* An Azure IoT Hub set up
* Raspberry Pi 3+ (with Raspbian Stretch and [Azure IoT Edge runtime](https://docs.microsoft.com/en-us/azure/iot-edge/how-to-install-iot-edge-linux-arm/?WT.mc_id=devto-blog-dglover) installed)
* PiCamera (with the 'Camera' option enabled on your Rpi 3)
* An Azure DevOps account

## Overview
This MLOps implementation is based on this [MLOPs template](https://github.com/microsoft/MLOpsPython) with the goal to implement pipelines that automate the process from the training of Computer Vision models to their deployment on IoT Edge devices. 

This implementation features the retraining of YOLOv3 on the VOC dataset as shown in the first section of the guide and targetting the Raspberry Pi 3B as device. 
Still, all the good practices are demonstrated and this implementation can be re-used as a template for other Computer Vision models as well !

<p align="center"><img width="80%" src="https://github.com/microsoft/azure-iot-edge-cv-model-samples/blob/master/Documentation/mlops.png" /></p>
