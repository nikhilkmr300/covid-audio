import numpy as np
import random
import os
import shutil
import sys
from tqdm import tqdm

# Seed to shuffle images randomly.
random.seed(1)

# Source directories (copy from)
data_dir_covid = 'data_clean/covid'     # Path to covid images in data_clean directory (i.e., before splitting).
data_dir_normal = 'data_clean/normal'   # Path to normal images in data_clean directory (i.e., before splitting).
# Destination directories (copy to)
train_dir_covid = 'train/covid'
train_dir_normal = 'train/normal'
test_dir_covid = 'test/covid'
test_dir_normal = 'test/normal'

# The sum of fractions need not add up to 1 if you plan to use only some part of
# the data.
# Split between covid and normal in train and test is done in the
# same ratio of covid to normal in data_clean.
train_frac = 0.8
test_frac = 0.2

filepaths_covid = list()    # All available covid images in data_clean directory.
filepaths_normal = list()   # All available normal images in data_clean directory.
for file in sorted(os.listdir(data_dir_covid)):
    _, extension = os.path.splitext(file)
    if extension == '.png':
        filepaths_covid.append(os.path.join(data_dir_covid, file))
for file in sorted(os.listdir(data_dir_normal)):
    _, extension = os.path.splitext(file)
    if extension == '.png':
        filepaths_normal.append(os.path.join(data_dir_normal, file))

print(f'Found {len(filepaths_covid)} images of class covid in {data_dir_covid}.')
print(f'Found {len(filepaths_normal)} images of class covid in {data_dir_normal}.')

random.shuffle(filepaths_covid)
random.shuffle(filepaths_normal)

# Making the split.
# Filepaths to images in **data_clean** to copy to train/covid.
train_paths_covid = filepaths_covid[:int(train_frac * len(filepaths_covid))]
# Filepaths to images in **data_clean** to copy to test/covid.
test_paths_covid = filepaths_covid[int(train_frac * len(filepaths_covid)):int((train_frac + test_frac) * len(filepaths_covid))]
# Filepaths to images in **data_clean** to copy to train/normal.
train_paths_normal = filepaths_normal[:int(train_frac * len(filepaths_normal))]
# Filepaths to images in **data_clean** to copy to test/normal.
test_paths_normal = filepaths_normal[int(train_frac * len(filepaths_normal)):int((train_frac + test_frac) * len(filepaths_normal))]

# Checking that the split maintains ratio of covid to normal in both train and
# test.
np.testing.assert_almost_equal(
    len(filepaths_covid) / len(filepaths_normal),
    len(train_paths_covid) / len(train_paths_normal),
    err_msg="covid:normal ratio in train is not equal to covid:normal in data",
    decimal=2
)
np.testing.assert_almost_equal(
    len(filepaths_covid) / len(filepaths_normal),
    len(test_paths_covid) / len(test_paths_normal),
    err_msg="covid:normal ratio in test is not equal to covid:normal in data",
    decimal=2
)

# Deleting train and test directories if they exist.
for dir in [train_dir_covid, train_dir_normal, test_dir_covid, test_dir_normal]:
    if os.path.exists(dir):
        shutil.rmtree(dir)
# Creating new train and test directories.
os.makedirs(train_dir_covid)
os.makedirs(train_dir_normal)
os.makedirs(test_dir_covid)
os.makedirs(test_dir_normal)

# train_frac of data_clean/covid to train/covid.
print(f'Copying {len(train_paths_covid)} [={train_frac}*{len(filepaths_covid)}] training images of class covid to {train_dir_covid}...')
for path in tqdm(train_paths_covid):
    basename = os.path.basename(path)
    shutil.copy(path, os.path.join(train_dir_covid, basename))
# train_frac of data_clean/normal to train/normal.
print(f'Copying {len(train_paths_normal)} [={train_frac}*{len(filepaths_normal)}] training images of class normal to {train_dir_normal}...')
for path in tqdm(train_paths_normal):
    basename = os.path.basename(path)
    shutil.copy(path, os.path.join(train_dir_normal, basename))
# test_frac of data_clean/covid to test/covid.
print(f'Copying {len(test_paths_covid)} [={test_frac}*{len(filepaths_covid)}] testing images of class covid to {test_dir_covid}...')
for path in tqdm(test_paths_covid):
    basename = os.path.basename(path)
    shutil.copy(path, os.path.join(test_dir_covid, basename))
# test_frac of data_clean/normal to test/normal.
print(f'Copying {len(test_paths_normal)} [={test_frac}*{len(filepaths_normal)}] testing images of class normal to {test_dir_normal}...')
for path in tqdm(test_paths_normal):
    basename = os.path.basename(path)
    shutil.copy(path, os.path.join(test_dir_normal, basename))
