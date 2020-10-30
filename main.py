import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
import tensorflow as tf
from keras_data_generator import keras_data_generator, double_input_data_generator
from tensorflow.keras.callbacks import ModelCheckpoint
from data_sort import data_sort
from InceptionResNetV2_mixed import InceptionResNetV2

image_path='D:/Connor/Kaggle/melanoma_detection/siim-isic-melanoma-classification/jpeg/train'
csv_path = 'D:/Connor/Kaggle/melanoma_detection/siim-isic-melanoma-classification/train.csv'
weights_folder = 'D:/Connor/Kaggle/melanoma_detection/models/InceptionResNetV2_mixed/weights'
batch_size = 4
dim = (299,299)
validate = False
val_ratio = 0.2

if validate == True:
    all_image_paths, all_image_labels, all_image_info, train_idx, val_idx = data_sort(image_path, csv_path, val_ratio)
    training_generator = double_input_data_generator(train_idx, all_image_paths, all_image_info, all_image_labels, batch_size=batch_size, dim=dim, augment=True)
    validation_generator = double_input_data_generator(val_idx, all_image_paths, all_image_info, all_image_labels, batch_size=batch_size, dim=dim, augment=False)
else:
    all_image_paths, all_image_labels, all_image_info, train_idx = data_sort(image_path, csv_path, 0)
    training_generator = double_input_data_generator(train_idx, all_image_paths, all_image_info, all_image_labels, batch_size=batch_size, dim=dim, augment=True)

def init_model():
    #load model
    model = InceptionResNetV2(dim)
    #model.summary()
    print('model loaded.')
    
    #compile model
    checkpoint = checkpoint = ModelCheckpoint("weights/weights_epoch_{epoch:03d}_loss_{loss:.4f}.hdf5", monitor='loss', verbose=0, save_best_only=False, save_freq='epoch', save_weights_only = True)
    callbacks = [checkpoint]
    model.compile(optimizer=tf.keras.optimizers.Adam(),loss=tf.keras.losses.BinaryCrossentropy(),metrics=['accuracy'])
    
    #load weights
    epoch_num = 0
    if len(os.listdir(weights_folder)) > 0:
        weights_file = os.path.join(weights_folder, os.listdir(weights_folder)[-1])
        if os.path.isfile(weights_file):
            epoch_num = int(os.path.basename(weights_file)[14:17])
            model.load_weights(weights_file)
            print('weights loaded. file name: ' + os.path.basename(weights_file))
        else:
            print('imagenet weights loaded.')
    else:
        print('imagenet weights loaded.')
    
    return model, callbacks, epoch_num

model, callbacks, epoch_num = init_model()

#train model with error handling
while True:
    try:
        if validate == True:
            model.fit(training_generator, epochs=100, validation_data=validation_generator, callbacks=callbacks, max_queue_size = 200, workers=20, initial_epoch=epoch_num)
        else:
            model.fit(training_generator, epochs=100, callbacks=callbacks, max_queue_size = 200, workers=20, initial_epoch=epoch_num)
    except:
        print('Error encountered. Resuming training... ')
        model, callbacks, epoch_num = init_model()