import dlib
import os

train_dir = os.path.join('..', 'train')
valid_dir = os.path.join('..', 'valid')
test_dir = os.path.join('..', 'test')

train_xml_path = os.path.join('..', 'data', 'dlib_annot', 'train_annot.xml')
valid_xml_path = os.path.join('..', 'data', 'dlib_annot', 'valid_annot.xml')
test_xml_path = os.path.join('..', 'data', 'dlib_annot', 'test_annot.xml')
model_path = 'ert1.dat'

face_detector = dlib.get
shape_predictor =
