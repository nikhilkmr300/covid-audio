import random
import os
import shutil
import sys
from tqdm import tqdm

# Seed to shuffle images randomly.
random.seed(1)

data_dir = 'FaceDB_Snapshot_complete'
train_dir = 'train'
valid_dir = 'valid'
test_dir = 'test'

# The sum of fractions need not add up to 1 if you plan to use only some part of
# the data.
train_frac = 0.8
valid_frac = 0.1
test_frac = 0.1

filepaths = list()
for file in sorted(os.listdir(data_dir)):
    _, extension = os.path.splitext(file)
    if extension == '.png':
        filepaths.append(os.path.join(data_dir, file))

print(f'Found {len(filepaths)} images in {data_dir}.')

random.shuffle(filepaths)

train_paths = filepaths[:int(train_frac * len(filepaths))]
valid_paths = filepaths[int(train_frac * len(filepaths)):int((train_frac + valid_frac) * len(filepaths))]
test_paths = filepaths[int((train_frac + valid_frac) * len(filepaths)):int((train_frac + valid_frac + test_frac) * len(filepaths))]

# Deleting train, valid and test directories if they exist.
if os.path.exists(train_dir):
    shutil.rmtree(train_dir)
if os.path.exists(valid_dir):
    shutil.rmtree(valid_dir)
if os.path.exists(test_dir):
    shutil.rmtree(test_dir)
# Creating new train, valid and test directories.
os.makedirs(train_dir)
os.makedirs(valid_dir)
os.makedirs(test_dir)

# path contains <img_id>.png. Annotation for an image with ID img_id is
# <img_id>.ljson. So to get the path to the annotation ljson file, we need to
# remove the .png extension and append the .ljson extension
print(f'Copying {len(train_paths)} [={train_frac}*{len(filepaths)}] training images and {len(train_paths)} annotations from {data_dir} to {train_dir}...')
for path in tqdm(train_paths):
    img_id = os.path.basename(path[:-4])
    img_basename = img_id + '.png'
    annot_basename = img_id + '.ljson'
    shutil.copy(os.path.join(data_dir, img_basename), os.path.join(train_dir, img_basename))
    shutil.copy(os.path.join(data_dir, annot_basename), os.path.join(train_dir, annot_basename))
print(f'Copying {len(valid_paths)} [={valid_frac}*{len(filepaths)}] validation images and {len(valid_paths)} annotations from {data_dir} to {valid_dir}...')
for path in tqdm(valid_paths):
    img_id = os.path.basename(path[:-4])
    img_basename = img_id + '.png'
    annot_basename = img_id + '.ljson'
    shutil.copy(os.path.join(data_dir, img_basename), os.path.join(valid_dir, img_basename))
    shutil.copy(os.path.join(data_dir, annot_basename), os.path.join(valid_dir, annot_basename))
print(f'Copying {len(test_paths)} [={test_frac}*{len(filepaths)}] test images and {len(test_paths)} annotations from {data_dir} to {test_dir}...')
for path in tqdm(test_paths):
    img_id = os.path.basename(path[:-4])
    img_basename = img_id + '.png'
    annot_basename = img_id + '.ljson'
    shutil.copy(os.path.join(data_dir, img_basename), os.path.join(test_dir, img_basename))
    shutil.copy(os.path.join(data_dir, annot_basename), os.path.join(test_dir, annot_basename))
