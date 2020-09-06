import numpy as np
import dlib
import time
import multiprocessing
import sys
import os

train_xml_path = os.path.join('..', 'data', 'dlib_annot', 'train_annot.xml')
valid_xml_path = os.path.join('..', 'data', 'dlib_annot', 'valid_annot.xml')
test_xml_path = os.path.join('..', 'data', 'dlib_annot', 'test_annot.xml')
model_path = 'ert1.dat'

response = input(f'Retraining will overwrite {model_path}. Are you sure you want to retrain {model_path[:-4]}? (y/n) ')
if response != 'y' and response !='Y':
    sys.exit()

# Hyperparameters
options = dlib.shape_predictor_training_options()
options.num_threads = multiprocessing.cpu_count()
options.be_verbose = True

print(f'Training model {model_path[:-4]} (this can take a while)...')
time_start = time.time()
dlib.train_shape_predictor(train_xml_path, model_path, options)
time_end = time.time()
print('Training done.')
print(f'Time to train = {round(time_end - time_start, 3)}s')
