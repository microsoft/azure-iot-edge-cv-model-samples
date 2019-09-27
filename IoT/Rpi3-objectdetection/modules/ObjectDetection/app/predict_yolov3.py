from PIL import Image
import numpy as np
import sys
import os
import numpy as np
import json
import onnxruntime

# Special json encoder for numpy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
            np.int16, np.int32, np.int64, np.uint8,
            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, 
            np.float64)):
            return float(obj)
        elif isinstance(obj,(np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

# Needed for preprocessing
def letterbox_image(image, size):
    iw, ih = image.size
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    return new_image

class YOLOv3Predict:
    def __init__(self, model, label_file):
        self.model = model
        self.label_file = label_file
        self.label = []

    def get_labels(self):
        with open(self.label_file) as f:
            for line in f:
                self.label.append(line.rstrip())

    def initialize(self):
        global session
        print('Loading model...')
        self.get_labels()
        session = onnxruntime.InferenceSession(self.model)
        print('Model loaded!')

    def preprocess(self,img):
        model_image_size = (416, 416)
        boxed_image = letterbox_image(img, tuple(reversed(model_image_size)))
        image_data = np.array(boxed_image, dtype='float32')
        image_data /= 255.
        image_data = np.transpose(image_data, [2, 0, 1])
        image_data = np.expand_dims(image_data, 0)
        return image_data

    def postprocess(self, boxes, scores, indices):
        out_boxes, out_scores, out_classes = [], [], []
        for idx_ in indices:
            out_classes.append(idx_[1])
            out_scores.append(scores[tuple(idx_)])
            idx_1 = (idx_[0], idx_[2])
            out_boxes.append(boxes[idx_1])
        return out_boxes, out_scores, out_classes

    def predict(self,image):
        image_data = self.preprocess(image)
        image_size = np.array([image.size[1], image.size[0]], dtype=np.int32).reshape(1, 2)
        input_names = session.get_inputs()
        feed_dict = {input_names[0].name: image_data, input_names[1].name: image_size}
        boxes, scores, indices = session.run([], input_feed=feed_dict)
        predicted_boxes, predicted_scores, predicted_classes = self.postprocess(boxes, scores, indices)
        results = []
        for i,c in enumerate(predicted_classes):
            data = {}
            data[self.label[c]] = json.dumps(predicted_boxes[i].tolist()+[predicted_scores[i]], cls=NumpyEncoder)
            results.append(data)
        return results
