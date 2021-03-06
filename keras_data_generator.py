import numpy as np
from random import randint
import cv2
from tensorflow.keras.utils import Sequence

class keras_data_generator(Sequence):
    """Generates data for Keras
    Sequence based data generator. Suitable for building data generator for training and prediction.
    """
    def __init__(self, list_IDs, image_filepaths, labels, to_fit=True, batch_size=32, dim=(1080, 1920), n_channels=3, n_classes=2, shuffle=False):
        """Initialization
        :param list_IDs: list of all 'image_filepaths' ids to use in the generator
        :param image_filepaths: list of image image_filepaths (file paths)
        :param labels: list of labels for images (int reffering to class)
        :param to_fit: True to return X and y, False to return X only
        :param batch_size: batch size at each iteration
        :param dim: tuple indicating image dimension
        :param n_channels: number of image channels
        :param n_classes: number of output masks
        :param shuffle: True to shuffle label indexes after every epoch
        """
        self.list_IDs = list_IDs
        self.image_filepaths = image_filepaths
        self.to_fit = to_fit
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.labels = labels
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch
        :return: number of batches per epoch
        """
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data
        :param index: index of the batch
        :return: X and y when fitting. X only when predicting
        """
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X = self._generate_X(list_IDs_temp)

        if self.to_fit:
            y = self._generate_y(list_IDs_temp)
            return X, y, [None]
        else:
            return X


    def on_epoch_end(self):
        """Updates indexes after each epoch
        """
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def _generate_X(self, list_IDs_temp):
        """Generates data containing batch_size images
        :param list_IDs_temp: list of label ids to load
        :return: batch of images
        """
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = self._load_colour_image(self.image_filepaths[ID])

        return X

    def _generate_y(self, list_IDs_temp):
        """Generates data containing batch_size masks
        :param list_IDs_temp: list of label ids to load
        :return: batch if masks
        """
        y = np.empty((self.batch_size, 1), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            y[i,] = self.labels[ID]

        return y

    def _load_colour_image(self, image_path):
        """Load grayscale image
        :param image_path: path to image to load
        :return: loaded image
        """
        img = cv2.imread(image_path)
        img = img / 255
        img = cv2.resize(img, (self.dim[1], self.dim[0])) 
        return img


class double_input_data_generator(Sequence):
    """Generates data for Keras
    Sequence based data generator. Suitable for building data generator for training and prediction.
    """
    def __init__(self, list_IDs, image_filepaths, info_lists, labels, to_fit=True, batch_size=32, dim=(1024, 1024), n_channels=3, n_classes=2, shuffle=False, augment=False):
        """Initialization
        :param list_IDs: list of all 'image_filepaths' ids to use in the generator
        :param image_filepaths: list of image image_filepaths (file paths)
        :param labels: list of labels for images (int reffering to class)
        :param to_fit: True to return X and y, False to return X only
        :param batch_size: batch size at each iteration
        :param dim: tuple indicating image dimension
        :param n_channels: number of image channels
        :param n_classes: number of output masks
        :param shuffle: True to shuffle label indexes after every epoch
        """
        self.list_IDs = list_IDs
        self.image_filepaths = image_filepaths
        self.info_lists = info_lists
        self.to_fit = to_fit
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.labels = labels
        self.augment = augment
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch
        :return: number of batches per epoch
        """
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data
        :param index: index of the batch
        :return: X and y when fitting. X only when predicting
        """
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X1 = self._generate_X1(list_IDs_temp)
        X2 = self._generate_X2(list_IDs_temp)

        if self.to_fit:
            y = self._generate_y(list_IDs_temp)
            return [X1, X2], y, [None]
        else:
            return [X1, X2]

    def on_epoch_end(self):

        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def _generate_X1(self, list_IDs_temp):

        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = self._load_colour_image(self.image_filepaths[ID])

        return X

    def _generate_y(self, list_IDs_temp):

        y = np.empty((self.batch_size, 1), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            y[i,] = self.labels[ID]

        return y
    
    def _generate_X2(self, list_IDs_temp):

        X2 = np.empty((self.batch_size, 48), dtype=float)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X2[i,] = self.info_lists[ID]

        return X2

    def _load_colour_image(self, image_path):

        img = cv2.imread(image_path)
        img = cv2.resize(img, (self.dim[1], self.dim[0]))
        if self.augment == True:
            #flips
            if randint(1,4) == 1:
                img = img
            else:
                img = cv2.flip(img, (2-randint(1,3)))  
            #brightness
            increase = float(100-randint(1,200))
            img = cv2.add(img,np.array([increase]))
        
        img = img / 255
        
        return img