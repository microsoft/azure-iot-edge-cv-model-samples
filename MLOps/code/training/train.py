"""
Retrain the YOLO model for your own dataset.
"""

# more imports for os operations
import argparse
import shutil
import os

import numpy as np
import keras.backend as K
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, Callback

from yolo3.model import preprocess_true_boxes, yolo_body, tiny_yolo_body, yolo_loss
from yolo3.utils import get_random_data
from voc_annotation import write_annotation

from convert_yolov3_to_onnx import YOLO, convert_model

# Fix issue "AttributeError: module 'keras.backend' has no attribute 'control_flow_ops" see https://github.com/keras-team/keras/issues/3857
import tensorflow as tf
K.control_flow_ops = tf

# Logging for azure ML
from azureml.core.run import Run

# Get run when running in remote
if 'run' not in locals():
    run = Run.get_context()

def _main(model_name, release_id, model_path, fine_tune_epochs, unfrozen_epochs, learning_rate):
    annotation_path = 'train.txt'
    log_dir = 'logs/'
    classes_path = model_path+'/classes.txt'
    anchors_path = model_path+'/yolo_anchors.txt'
    class_names = get_classes(classes_path)
    num_classes = len(class_names)
    anchors = get_anchors(anchors_path)

    input_shape = (416,416) # multiple of 32, hw

    is_tiny_version = len(anchors)==6 # default setting
    if is_tiny_version:
        model = create_tiny_model(input_shape, anchors, num_classes,
            freeze_body=2, weights_path=model_path+'/tiny_yolo_weights.h5')
    else:
        model = create_model(input_shape, anchors, num_classes,
            freeze_body=2, weights_path=model_path+'/yolo_weights.h5') # make sure you know what you freeze
    
    # Define callbacks during training
    logging = TensorBoard(log_dir=log_dir)
    checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
        monitor='val_loss', save_weights_only=True, save_best_only=True, period=3)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

    # Logging for Azure ML (send acc, loss, val_loss at the end of each epoch)
    class LossHistory1(Callback):
        def on_epoch_end(self, epoch, logs={}):
            run.log('Loss_stage1', logs.get('loss'))
            run.log('Val_Loss_stage1', logs.get('val_loss'))

    class LossHistory2(Callback):
        def on_epoch_end(self, epoch, logs={}):
            run.log('Loss_stage2', logs.get('loss'))
            run.log('Val_Loss_stage2', logs.get('val_loss'))
    
    lossHistory1 = LossHistory1()
    lossHistory2 = LossHistory2()

    val_split = 0.2
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines)*val_split)
    num_train = len(lines) - num_val

    # (Stage 1) Train with frozen layers first, to get a stable loss.
    # Adjust num epochs to your dataset. This step is enough to obtain a not bad model.
    if True:
        model.compile(optimizer=Adam(lr=learning_rate), loss={
            # use custom yolo_loss Lambda layer.
            'yolo_loss': lambda y_true, y_pred: y_pred})

        batch_size = 50
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
                steps_per_epoch=max(1, num_train//batch_size),
                validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes),
                validation_steps=max(1, num_val//batch_size),
                epochs=fine_tune_epochs,
                initial_epoch=0,
                callbacks=[logging, checkpoint, lossHistory1])
        model.save_weights(log_dir + model_name)

    # (Stage 2) Unfreeze and continue training, to fine-tune.
    # Train longer if the result is not good.
    if True:
        for i in range(len(model.layers)):
            model.layers[i].trainable = True
        model.compile(optimizer=Adam(lr=learning_rate), loss={'yolo_loss': lambda y_true, y_pred: y_pred}) # recompile to apply the change
        print('Unfreeze all of the layers.')

        batch_size = 2 # note that more GPU memory is required after unfreezing the body
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
            steps_per_epoch=max(1, num_train//batch_size),
            validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes),
            validation_steps=max(1, num_val//batch_size),
            epochs=unfrozen_epochs,
            initial_epoch=fine_tune_epochs,
            callbacks=[logging, checkpoint, reduce_lr, early_stopping, lossHistory2])
        model.save_weights(log_dir + model_name)

    # Further training if needed...

    # Add properties to identify this specific training run
    run.add_properties({"release_id": release_id, "run_type": "train"})
    print(f"added properties: {run.properties}")

    try:
        model_name_path = os.path.join(log_dir, model_name)
        print(model_name_path)

        new_model_name =  model_name.rstrip('.h5')+'.onnx'
        convert_model(YOLO(model_name_path, model_path), log_dir + new_model_name, 10)

        new_model_path = os.path.join(log_dir, new_model_name)
        print(new_model_path)
        run.register_model(
            model_name=new_model_name,
            model_path=new_model_path,
            properties={"release_id": release_id})
        print("Registered new model!")
    except Exception as e:
        print(e)

    print(run.get_file_names())

    run.complete()


def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)

def create_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2,
            weights_path='model_data/yolo_weights.h5'):
    '''create the training model'''
    K.clear_session() # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    y_true = [Input(shape=(h//{0:32, 1:16, 2:8}[l], w//{0:32, 1:16, 2:8}[l], \
        num_anchors//3, num_classes+5)) for l in range(3)]

    model_body = yolo_body(image_input, num_anchors//3, num_classes)
    print('Create YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        if freeze_body in [1, 2]:
            # Freeze darknet53 body or freeze all but 3 output layers.
            num = (185, len(model_body.layers)-3)[freeze_body-1]
            for i in range(num): model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})(
        [*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)

    return model

def create_tiny_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2,
            weights_path='model_data/tiny_yolo_weights.h5'):
    '''create the training model, for Tiny YOLOv3'''
    K.clear_session() # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    y_true = [Input(shape=(h//{0:32, 1:16}[l], w//{0:32, 1:16}[l], \
        num_anchors//2, num_classes+5)) for l in range(2)]

    model_body = tiny_yolo_body(image_input, num_anchors//2, num_classes)
    print('Create Tiny YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        if freeze_body in [1, 2]:
            # Freeze the darknet body or freeze all but 2 output layers.
            num = (20, len(model_body.layers)-2)[freeze_body-1]
            for i in range(num): model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.7})(
        [*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)

    return model

def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes):
    '''data generator for fit_generator'''
    n = len(annotation_lines)
    i = 0
    while True:
        image_data = []
        box_data = []
        for b in range(batch_size):
            if i==0:
                np.random.shuffle(annotation_lines)
            image, box = get_random_data(annotation_lines[i], input_shape, random=True)
            image_data.append(image)
            box_data.append(box)
            i = (i+1) % n
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        yield [image_data, *y_true], np.zeros(batch_size)

def data_generator_wrapper(annotation_lines, batch_size, input_shape, anchors, num_classes):
    n = len(annotation_lines)
    if n==0 or batch_size<=0: return None
    return data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', type=str, help='Folder path for input data')
    parser.add_argument('--model_path', type=str, help='Folder path for model files')
    parser.add_argument('--chkpoint_folder', type=str, default='./logs', help='Folder path for checkpoint files')
    parser.add_argument('--fine_tune_epochs', type=int, default=50, help='Number of epochs for fine-tuning')
    parser.add_argument('--unfrozen_epochs', type=int, default=55, help='Final epoch for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    # Added for MLOps
    parser.add_argument("--release_id", type=str, help="The ID of the release triggering this pipeline run")
    parser.add_argument("--model_name", type=str, help="Name of the Model", default="yolo.h5",)

    FLAGS, unparsed = parser.parse_known_args()

    # Clean checkpoint folder if exists
    if os.path.exists(FLAGS.chkpoint_folder) :
        for file_name in os.listdir(FLAGS.chkpoint_folder):
            file_path = os.path.join(FLAGS.chkpoint_folder, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)

    # Write annotation file            
    write_annotation(FLAGS.data_folder)

    _main(FLAGS.model_name, FLAGS.release_id, FLAGS.model_path, FLAGS.fine_tune_epochs, FLAGS.unfrozen_epochs, FLAGS.learning_rate)
