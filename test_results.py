print('Evaluating models.')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import csv
from keras_data_generator import double_input_data_generator
from data_sort import data_sort
from InceptionResNetV2_mixed import InceptionResNetV2
import time
import threading
from threading import Thread

image_path='D:/Connor/Kaggle/melanoma_detection/siim-isic-melanoma-classification/jpeg/test'
csv_path = 'D:/Connor/Kaggle/melanoma_detection/siim-isic-melanoma-classification/test.csv'
weights_folder = 'D:/Connor/Kaggle/melanoma_detection/models/InceptionResNetV2_mixed/weights'
num_results = str(len(os.listdir('test_results/')))

dim = (299,299)
model_name = 'inceptionresnetv2_concat_'



class evaluation_stream:
    def __init__(self):
        self.stopped = False
        self.samples_processed = 0
        
    def start(self, weights_file, thread_index, model):
        self.model = model
        self.weights_file = weights_file
        self.thread_index = thread_index
        self.weights_file = os.path.join(weights_folder, weights_file)
        self.test_image_paths, self.test_image_labels, self.test_image_info, self.train_idx = data_sort(image_path, csv_path, 0, False)
        self.prediction_generator = double_input_data_generator(self.train_idx, self.test_image_paths, self.test_image_info, self.test_image_labels, batch_size=1, dim=dim, augment=False, to_fit=False)
        self.num_samples = len(self.test_image_paths)
        self.epoch_num = int(os.path.basename(self.weights_file)[14:17])
        self.test_results_csv_path = 'test_results/test_results_model_' + model_name + 'epoch' + str(self.epoch_num) + '.csv'
        self.model.load_weights(self.weights_file)
        print('Thread number: ' + str(self.thread_index) + '. weights loaded. file name: ' + os.path.basename(self.weights_file) + '. ' + str(self.num_samples) + ' samples to test.')
        self.file = open(self.test_results_csv_path, 'a', newline='')
        self.wr = csv.writer(self.file)
        self.wr.writerow(['image_name','target'])
        self.file.close
        self.t = Thread(target=self.predict, args=())
        self.t.daemon = True
        self.t.start()
        return self
    
    def predict(self):

        while self.stopped == False:
            for self.i in range(self.num_samples+1):
                #print('\r' + 'Testing sample number:' + str(self.i) + '. ' + str(int((self.i*100)/self.num_samples)) + '% Complete.   ', end="")
                if self.i < self.num_samples:
                    self.next_image = self.prediction_generator.__getitem__(self.i)
                    self.prediction = self.model.predict(self.next_image)
                    self.prediction = float(self.prediction)
                    self.test_image_labels[self.i][1] = self.prediction
                    self.file = open(self.test_results_csv_path, 'a', newline='')
                    self.wr = csv.writer(self.file)
                    self.wr.writerow(self.test_image_labels[self.i])
                    self.file.close
                    self.samples_processed = self.samples_processed + 1
                else:
                    self.file = open(self.test_results_csv_path, 'a', newline='')
                    self.wr = csv.writer(self.file)
                    self.wr.writerow(['end','0'])
                    self.file.close
                    self.samples_processed = self.samples_processed + 1
                    self.stop()
    
    def stop(self):
	    self.stopped = True



weights_file_list = os.listdir(weights_folder)
evaluation_streams = []
models = []
num_done = len(os.listdir('D:/Connor/Kaggle/melanoma_detection/models/InceptionResNetV2_mixed/test_results/'))
for j, weights_file in enumerate(weights_file_list[num_done:]):
    models.append(InceptionResNetV2(dim))
    print('Model ' + str(j) + ' loaded.')
    evaluation_streams.append(evaluation_stream().start(weights_file, j, models[j]))

complete = False
while complete == False:
    complete = True
    total = 0
    for i in evaluation_streams:
        total = total + i.samples_processed
        if i.stopped == False:
            complete = False
    percentage = round((total/(i.num_samples*len(evaluation_streams)))*100, 2)
    print('\r' + 'Inference on ' + str(len(evaluation_streams)) + ' models. ' + str(percentage) + '% Complete.', end = '')
