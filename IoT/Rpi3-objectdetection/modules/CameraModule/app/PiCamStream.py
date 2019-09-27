import io
import time
import threading
import json
import requests

class ImageProcessor(threading.Thread):
    def __init__(self, owner):
        super(ImageProcessor, self).__init__()
        self.stream = io.BytesIO()
        self.event = threading.Event()
        self.terminated = False
        self.owner = owner
        self.start()

    def sendFrameForProcessing(self):
        """send a POST request to the AI module server"""
        headers = {'Content-Type': 'application/octet-stream'}
        try:
            self.stream.seek(0)
            response = requests.post(self.owner.endPointForProcessing, headers = headers, data = self.stream)
        except Exception as e:
            print('sendFrameForProcessing Exception -' + str(e))
            return "[]"
    
        return json.dumps(response.json())

    def run(self):
        # This method runs in a separate thread
        while not self.terminated:
            # Wait for an image to be written to the stream
            if self.event.wait(1):
                try:
                    result = self.sendFrameForProcessing()
                    if result != "[]":
                        self.owner.sendToHubCallback(result) # send message to the hub if an object has been detected
                    print(result)
                    #self.owner.done=True 
                    # uncomment above if you want the process to terminate for some reason
                finally:
                    # Reset the stream and event
                    self.stream.seek(0)
                    self.stream.truncate()
                    self.event.clear()
                    # Return ourselves to the available pool
                    with self.owner.lock:
                        self.owner.pool.append(self)

class ProcessOutput(object):
    def __init__(self, endPoint, functionCallBack):
        self.done = False
        # Construct a pool of 4 image processors along with a lock
        # to control access between threads
        # Note that you can vary the number depending on how many processors your device has
        self.lock = threading.Lock()
        self.endPointForProcessing = endPoint
        self.sendToHubCallback = functionCallBack
        self.pool = [ImageProcessor(self) for i in range(4)]
        self.processor = None

    def write(self, buf):
        if buf.startswith(b'\xff\xd8'):
            # New frame; set the current processor going and grab
            # a spare one
            if self.processor:
                self.processor.event.set()
            with self.lock:
                if self.pool:
                    self.processor = self.pool.pop()
                else:
                    # No processor's available, we'll have to skip
                    # this frame; you may want to print a warning
                    # here to see whether you hit this case
                    self.processor = None
        if self.processor:
            self.processor.stream.write(buf)

    def flush(self):
        # When told to flush (this indicates end of recording), shut
        # down in an orderly fashion. First, add the current processor
        # back to the pool
        if self.processor:
            with self.lock:
                self.pool.append(self.processor)
                self.processor = None
        # Now, empty the pool, joining each thread as we go
        while True:
            with self.lock:
                try:
                    proc = self.pool.pop()
                except IndexError:
                    pass # pool is empty
            proc.terminated = True
            proc.join()