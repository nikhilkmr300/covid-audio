import dlib
import cv2
import time
import shutil
import os
from tqdm import tqdm

valid_dir = os.path.join('..', 'data', 'valid')
results_dir = os.path.join('..', 'results_face', 'results_cnn')

# Deleting directory if it exists.
if os.path.exists(results_dir):
    shutil.rmtree(results_dir)
# Creating directory to store predictions of CNN.
os.makedirs(results_dir)

face_detector = dlib.cnn_face_detection_model_v1('mmod_human_face_detector.dat')

print(f'Generating results of CNN on validation images, writing resulting images to {results_dir}...')
correct = 0 # Number of bounding boxes correctly generated for images in validation.
count = 0   # Total number of images in validation.
time_predict = 0
for basename in tqdm(sorted(os.listdir(valid_dir))):
    path = os.path.join(valid_dir, basename)
    if(path[-4:] == '.png'):
        count += 1
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
            img = cv2.rectangle(img, (box.rect.left(), box.rect.top()), (box.rect.right(), box.rect.bottom()), (255, 0, 0), 3)
        # Writing image to results_dir.
        cv2.imwrite(os.path.join(results_dir, basename), img)

print(f'Average prediction time per image = {round(time_predict / count, 3)}s')
print(f'Correctly detected: {correct}/{count}')
