# YOLOv3 Object Detection on RPi 3 with Azure IoT Edge
This is an IoT solution based on Azure IoT Edge that can perform object detection (YOLOv3) on a Raspberry Pi 3 equipped with the PiCamera v2.

## Prerequisites
* An [Azure account](https://account.microsoft.com/account?lang=en-us)
* An Azure subscription
* An Azure IoT Hub set up (you can quickly configure one using ../01-configure-iot-hub.ipynb)
* Raspberry Pi 3+ (with Raspbian Stretch and [Azure IoT Edge runtime](https://docs.microsoft.com/en-us/azure/iot-edge/how-to-install-iot-edge-linux-arm/?WT.mc_id=devto-blog-dglover) installed)
* PiCamera (with the 'Camera' option enabled on your Rpi 3)

## Getting started
Clone this repo, to test on your machine, you need to download a [YOLOv3 model](https://onnxzoo.blob.core.windows.net/models/opset_10/yolov3/yolov3.onnx), rename it 'model.onnx' and move it inside the (modules > ObjectDetection >) app/ folder.
If you use your own trained YOLOv3 model, you will also need to update the labels.txt file.

## (Quick) Building and deploying the solution
You can either follow the guide to set up the building and the deployment or if you are interested in testing the solution, run the following commands (you will need to have [Azure CLI](https://docs.microsoft.com/en-us/cli/azure/?view=azure-cli-latest) installed):
```
az login
az iot edge set-modules --device-id [device id] --hub-name [hub name] --content deployment.arm32v7.json
```
(Don't forget to change the [device id] and the [hub name] with your own device id and hub name)

You need to wait a couple minutes for it to pull all the docker modules and run it on your Raspberry Pi.

## Building and deploying your own solution
If you want your docker modules to be on your own private container registries, open up the .env file and specify your registry credentials:
```
CONTAINER_REGISTRY_ADDRESS= ...
CONTAINER_REGISTRY_USERNAME= ...
CONTAINER_REGISTRY_PASSWORD= ...
```
Then follow the guide for more infos on how to deploy it on your RPi 3.