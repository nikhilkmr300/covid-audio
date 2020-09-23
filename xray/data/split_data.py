import numpy as np
import random
import os
import shutil
import sys
from tqdm import tqdm

# Extensions for images in data_clean directory.
EXTENSIONS = {'.png'}

# Seed to shuffle images randomly.
random.seed(1)

# Source directories (copy from)
data_dir_covid = 'data_clean/covid'     # Path to covid images in data_clean directory (i.e., before splitting).
data_dir_normal = 'data_clean/normal'   # Path to normal images in data_clean directory (i.e., before splitting).
# Destination directories (copy to)
train_dir_covid = 'train/covid'
train_dir_normal = 'train/normal'
valid_dir_covid = 'valid/covid'
valid_dir_normal = 'valid/normal'
test_dir_covid = 'test/covid'
test_dir_normal = 'test/normal'

# The sum of fractions need not add up to 1 if you plan to use only some part of
# the data.
train_frac = 0.8
valid_frac = 0.1
test_frac = 0.1

# Ratio of normal to covid in train, valid and test. Choose 1 for perfectly
# balanced split. Ensure you provide a ratio that is possible with the data.
normal_to_covid = 1.5

filepaths_covid = list()    # All available covid images in data_clean directory.
filepaths_normal = list()   # All available normal images in data_clean directory.
for file in sorted(os.listdir(data_dir_covid)):
    _, extension = os.path.splitext(file)
    if extension in EXTENSIONS:
        filepaths_covid.append(os.path.join(data_dir_covid, file))
for file in sorted(os.listdir(data_dir_normal)):
    _, extension = os.path.splitext(file)
    if extension in EXTENSIONS:
        filepaths_normal.append(os.path.join(data_dir_normal, file))

print(f'Found {len(filepaths_covid)} images of class covid in {data_dir_covid}.')
print(f'Found {len(filepaths_normal)} images of class covid in {data_dir_normal}.')

random.shuffle(filepaths_covid)
random.shuffle(filepaths_normal)

print(f'normal:covid ratio (desired) = {normal_to_covid} (you can change this value in split_data.py)')
# Making the split. Split is done in such a way that whatever the ratio of
# normal to covid normal_to_covid, all the covid images are always included.
# Filepaths to images in **data_clean** to copy to train/covid.
train_paths_covid = filepaths_covid[:int(train_frac * len(filepaths_covid))]
# Filepaths to images in **data_clean** to copy to valid/covid.
valid_paths_covid = filepaths_covid[int(train_frac * len(filepaths_covid)):int((train_frac + valid_frac) * len(filepaths_covid))]
# Filepaths to images in **data_clean** to copy to test/covid.
test_paths_covid = filepaths_covid[int((train_frac + valid_frac) * len(filepaths_covid)):int((train_frac + valid_frac + test_frac) * len(filepaths_covid))]
# Filepaths to images in **data_clean** to copy to train/normal. Contains
# normal_to_covid * num_covid_images number of normal images.
train_paths_normal = filepaths_normal[:int(normal_to_covid * train_frac * len(filepaths_covid))]
# Filepaths to images in **data_clean** to copy to valid/normal. Contains
# normal_to_covid * num_covid_images number of normal images.
valid_paths_normal = filepaths_normal[int(normal_to_covid * train_frac * len(filepaths_covid)):int(normal_to_covid * (train_frac + valid_frac) * len(filepaths_covid))]
# Filepaths to images in **data_clean** to copy to test/normal. Contains
# normal_to_covid * num_covid_images number of normal images.
test_paths_normal = filepaths_normal[int(normal_to_covid * (train_frac + valid_frac) * len(filepaths_covid)):int(normal_to_covid * (train_frac + valid_frac + test_frac) * len(filepaths_covid))]

# Checking that the split maintains the promised normal_to_covid ratio in train.
err_msg = "Given normal:covid ratio is not possible with the data (not enough images of 'normal' class). Try reducing the normal:covid ratio."
np.testing.assert_almost_equal(
    len(train_paths_normal) / len(train_paths_covid),
    normal_to_covid,
    err_msg=err_msg,
    decimal=2
)
# Checking that the split maintains the promised normal_to_covid ratio in valid.
np.testing.assert_almost_equal(
    len(valid_paths_normal) / len(valid_paths_covid),
    normal_to_covid,
    err_msg=err_msg,
    decimal=2
)
# Checking that the split maintains the promised normal_to_covid ratio in test.
np.testing.assert_almost_equal(
    len(test_paths_normal) / len(test_paths_covid),
    normal_to_covid,
    err_msg=err_msg,
    decimal=2
)

# Deleting train, valid and test directories if they exist.
for dir in [train_dir_covid, train_dir_normal, valid_dir_covid, valid_dir_normal, test_dir_covid, test_dir_normal]:
    if os.path.exists(dir):
        shutil.rmtree(dir)
# Creating new train, valid and test directories.
os.makedirs(train_dir_covid)
os.makedirs(train_dir_normal)
os.makedirs(valid_dir_covid)
os.makedirs(valid_dir_normal)
os.makedirs(test_dir_covid)
os.makedirs(test_dir_normal)

# data_clean/covid to train/covid.
print(f'Copying {len(train_paths_covid)} [={train_frac}*{len(filepaths_covid)}] training images of class covid to {train_dir_covid}...')
for path in tqdm(train_paths_covid):
    basename = os.path.basename(path)
    shutil.copy(path, os.path.join(train_dir_covid, basename))
# data_clean/normal to train/normal.
print(f'Copying {len(train_paths_normal)} [={normal_to_covid}*{train_frac}*{len(filepaths_covid)}] training images of class normal to {train_dir_normal}...')
for path in tqdm(train_paths_normal):
    basename = os.path.basename(path)
    shutil.copy(path, os.path.join(train_dir_normal, basename))
# data_clean/covid to valid/covid.
print(f'Copying {len(valid_paths_covid)} [={valid_frac}*{len(filepaths_covid)}] validation images of class covid to {valid_dir_covid}...')
for path in tqdm(valid_paths_covid):
    basename = os.path.basename(path)
    shutil.copy(path, os.path.join(valid_dir_covid, basename))
# data_clean/normal to valid/normal.
print(f'Copying {len(valid_paths_normal)} [={normal_to_covid}*{valid_frac}*{len(filepaths_covid)}] validation images of class normal to {valid_dir_normal}...')
for path in tqdm(valid_paths_normal):
    basename = os.path.basename(path)
    shutil.copy(path, os.path.join(valid_dir_normal, basename))
# data_clean/covid to test/covid.
print(f'Copying {len(test_paths_covid)} [={test_frac}*{len(filepaths_covid)}] testing images of class covid to {test_dir_covid}...')
for path in tqdm(test_paths_covid):
    basename = os.path.basename(path)
    shutil.copy(path, os.path.join(test_dir_covid, basename))
# data_clean/normal to test/normal.
print(f'Copying {len(test_paths_normal)} [={normal_to_covid}*{test_frac}*{len(filepaths_covid)}] testing images of class normal to {test_dir_normal}...')
for path in tqdm(test_paths_normal):
    basename = os.path.basename(path)
    shutil.copy(path, os.path.join(test_dir_normal, basename))
