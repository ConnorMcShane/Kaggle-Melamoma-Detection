import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Concatenate, Dropout

def InceptionResNetV2(dim, dropout=False):
    change_input = True
    if dim == (299,299):
        change_input = False
    with tf.device('/gpu:0'):
        inputs_1 = Input(shape=(dim[0], dim[1], 3))
        if change_input == True:
            model = tf.keras.applications.InceptionResNetV2(include_top=False, weights='imagenet', input_tensor=inputs_1, input_shape=(dim[0], dim[1], 3))
        else:
            model = tf.keras.applications.InceptionResNetV2(include_top=False, weights='imagenet', input_tensor=inputs_1)
        InceptionResNetV2_output = model.output
        flatten_1 = Flatten()(InceptionResNetV2_output)
        if dropout == True:
           flatten_1 = Dropout(0.2)(flatten_1)
        inputs_2 = Input(shape=(48))
        dense_1 = Dense(128, activation='relu')(inputs_2)
        dense_2 = Dense(128, activation='relu')(dense_1)
        if dropout == True:
           dense_2 = Dropout(0.2)(dense_2)
        concat_1 = Concatenate(axis=1)([flatten_1, dense_2])
        dense_3 = Dense(128, activation='relu')(concat_1)
        dense_4 = Dense(128, activation='relu')(dense_3)
        if dropout == True:
           dense_4 = Dropout(0.2)(dense_4)
        output = Dense(1, activation='sigmoid')(dense_4)
        model = Model(inputs=(inputs_1, inputs_2), outputs=output)
    return model