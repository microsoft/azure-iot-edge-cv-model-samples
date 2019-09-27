# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for
# full license information.

import random
import time
import sys

# use env variables
import os

import iothub_client
# pylint: disable=E0611
from iothub_client import IoTHubModuleClient, IoTHubClientError, IoTHubTransportProvider
from iothub_client import IoTHubMessage, IoTHubMessageDispositionResult, IoTHubError

# picamera imports
import PiCamStream
from PiCamStream import ProcessOutput
from picamera import PiCamera

# messageTimeout - the maximum time in milliseconds until a message times out.
# The timeout period starts at IoTHubModuleClient.send_event_async.
# By default, messages do not expire.
MESSAGE_TIMEOUT = 10000

# global counters
SEND_CALLBACKS = 0

# Choose HTTP, AMQP or MQTT as transport protocol.  Currently only MQTT is supported.
PROTOCOL = IoTHubTransportProvider.MQTT

# Send message to the Hub and forwards to the "output1" queue
def send_to_Hub_callback(strMessage):
    message = IoTHubMessage(bytearray(strMessage, 'utf8'))
    hubManager.send_event_to_output("output1", message, 0)

# Callback received when the message that we're forwarding is processed
def send_confirmation_callback(message, result, user_context):
    global SEND_CALLBACKS
    print ( "Confirmation[%d] received for message with result = %s" % (user_context, result) )
    map_properties = message.properties()
    key_value_pair = map_properties.get_internals()
    print ( "    Properties: %s" % key_value_pair )
    SEND_CALLBACKS += 1
    print ( "    Total calls confirmed: %d" % SEND_CALLBACKS )

class HubManager(object):

    def __init__(
            self,
            messageTimeout,
            protocol):
        self.client_protocol = protocol
        self.client = IoTHubModuleClient()
        self.client.create_from_environment(protocol)

        # set the time until a message times out
        self.client.set_option("messageTimeout", messageTimeout)

    # Forwards the message received onto the next stage in the process.
    def send_event_to_output(self, outputQueueName, event, send_context):
        self.client.send_event_async(
            outputQueueName, event, send_confirmation_callback, send_context)

def main(messageTimeout, protocol, imageProcessingEndpoint=""):
    try:
        print ( "\nPython %s\n" % sys.version )
        print ( "PiCamera module running. Press CTRL+C to exit" )
        try:
            global hubManager
            hubManager = HubManager(messageTimeout, protocol)
        except IoTHubError as iothub_error:
            print ( "Unexpected error %s from IoTHub" % iothub_error )
            return
        with PiCamera(resolution='VGA') as camera:
            camera.start_preview()
            time.sleep(2)
            output = ProcessOutput(imageProcessingEndpoint, send_to_Hub_callback)
            camera.start_recording(output, format='mjpeg')
            while not output.done:
                camera.wait_recording(1)
            camera.stop_recording()
    except KeyboardInterrupt:
        print ("PiCamera module stopped")

if __name__ == '__main__':
    try:
        IMAGE_PROCESSING_ENDPOINT = os.getenv('IMAGE_PROCESSING_ENDPOINT', "")
    except ValueError as error:
        print ( error )
        sys.exit(1)

    main(MESSAGE_TIMEOUT, PROTOCOL, IMAGE_PROCESSING_ENDPOINT)