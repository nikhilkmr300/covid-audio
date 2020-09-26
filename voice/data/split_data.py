import numpy as np
import random
import os
import shutil
import sys
from tqdm import tqdm

# Seed to shuffle images randomly.
random.seed(1)

# The sum of fractions need not add up to 1 if you plan to use only some part of
# the data. Fractions are for each audio type, independently.
train_frac = 0.8
valid_frac = 0.1
test_frac = 0.1

# audio_types contains the types of audio samples -- breath, cough.
audio_types = os.listdir('data_clean')

# Removing directories for audio_types if they already exist.
for audio_type in audio_types:
    path = 'data_' + audio_type
    if os.path.exists(path):
        shutil.rmtree(path)
# Making directories for each audio type.
for audio_type in audio_types:
    os.makedirs(os.path.join('data_' + audio_type, 'train'))
    os.makedirs(os.path.join('data_' + audio_type, 'valid'))
    os.makedirs(os.path.join('data_' + audio_type, 'test'))

# Performing train-valid-test split for each audio type.
for audio_type in audio_types:
    print(f'Performing train-valid-test split for {audio_type}...')
    # Source directory (copy from)
    source_path = os.path.join('data_clean', audio_type)
    # Destination directories (copy to)
    dest_path_train = os.path.join('data_' + audio_type, 'train')
    dest_path_valid = os.path.join('data_' + audio_type, 'valid')
    dest_path_test = os.path.join('data_' + audio_type, 'test')

    # Classes for the classification task (like asthma, covid, normal).
    classes = os.listdir(source_path)

    # Making train, valid and test destination directories for each class.
    for class_ in classes:
        os.makedirs(os.path.join(dest_path_train, class_))
        os.makedirs(os.path.join(dest_path_valid, class_))
        os.makedirs(os.path.join(dest_path_test, class_))

    # Filepaths to audio samples for each class.
    filepaths_classwise = dict()
    for class_ in classes:
        # Sorting as os.listdir gives a random order.
        paths = sorted(os.listdir(os.path.join(source_path, class_)))
        # Shuffling filepaths randomly to do away with any ordering.
        random.shuffle(paths)
        # Adding path from current directory before basename of file.
        paths = [os.path.join(source_path, class_, path) for path in paths]
        filepaths_classwise[class_] = paths

    # Making the split in such a way that proportion of classes remain same in
    # train, valid and test as they are in data_clean.
    for class_ in classes:
        print(f'\tFound {len(filepaths_classwise[class_])} audio samples of class {class_} in {source_path}.')

        # Paths for the current class of the current audio type.
        paths = filepaths_classwise[class_]
        train_paths = paths[:int(train_frac * len(paths))]
        valid_paths = paths[int(train_frac * len(paths)):int((train_frac + valid_frac) * len(paths))]
        test_paths = paths[int((train_frac + valid_frac) * len(paths)):int((train_frac + valid_frac + test_frac) * len(paths))]

        # Copying train_paths files to data_<audio_type>/train/<class>.
        print(f'\t\tCopying {len(train_paths)} [={train_frac}*{len(filepaths_classwise[class_])}] audio samples of class {class_} to {os.path.join(dest_path_train, class_)}...')
        for train_path in train_paths:
            shutil.copy(train_path, os.path.join(dest_path_train, class_))
        # Copying valid_paths files to data_<audio_type>/valid/<class>.
        print(f'\t\tCopying {len(valid_paths)} [={valid_frac}*{len(filepaths_classwise[class_])}] audio samples of class {class_} to {os.path.join(dest_path_valid, class_)}...')
        for valid_path in valid_paths:
            shutil.copy(valid_path, os.path.join(dest_path_valid, class_))
        # Copying test_paths files to data_<audio_type>/test/<class>.
        print(f'\t\tCopying {len(test_paths)} [={test_frac}*{len(filepaths_classwise[class_])}] audio samples of class {class_} to {os.path.join(dest_path_test, class_)}...')
        for test_path in test_paths:
            shutil.copy(test_path, os.path.join(dest_path_test, class_))
