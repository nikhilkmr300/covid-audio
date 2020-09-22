import dlib
import cv2
import time
import shutil
import os
from tqdm import tqdm

# Image extensions for images in valid.
EXTENSIONS = {'.png'}

valid_dir = os.path.join('..', 'data', 'valid')
results_dir = os.path.join('..', 'results_face', 'results_hog')

# Deleting directory if it exists.
if os.path.exists(results_dir):
    shutil.rmtree(results_dir)
# Creating directory to store predictions of HOG+SVM.
os.makedirs(results_dir)

face_detector = dlib.get_frontal_face_detector()

print(f'Generating results of HOG+SVM on validation images, writing resulting images to {results_dir}...')
correct = 0 # Number of bounding boxes correctly generated for images in validation.
time_predict = 0

# Validation image filenames.
valid_paths = os.listdir(valid_dir)
# Full paths to validation images.
valid_paths = [os.path.join(valid_dir, valid_path) for valid_path in valid_paths if valid_path[-4:] in EXTENSIONS]
for path in tqdm(sorted(valid_paths)):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    # Generating bounding boxes.
    time_start = time.time()
    boxes = face_detector(img)
    time_predict += time.time() - time_start
    # There should be exactly one face detected in each image.
    if len(boxes) == 1:
        correct += 1
    # Drawing boxes on image.
    for box in boxes:
        img = cv2.rectangle(img, (box.left(), box.top()), (box.right(), box.bottom()), (255, 0, 0), 3)
    # Writing image to results_dir.
    basename = os.path.basename(path)
    cv2.imwrite(os.path.join(results_dir, basename), img)

print(f'Average prediction time per image = {round(time_predict / len(valid_paths), 3)}s')
print(f'Correctly detected: {correct}/{len(valid_paths)}')
