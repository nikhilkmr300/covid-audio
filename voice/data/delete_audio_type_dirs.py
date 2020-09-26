# Deletes the directories for each type of audio sample -- breath and cough
# (for now, but generalizing in case other types of audio sample are
# available in the future). Directories that will be deleted have name of the
# form data_<audio_type>, except data_clean. Each data_<class> directory
# contains train, valid and test directories.

import os
import shutil
import glob
import sys

dirs_to_delete = glob.glob('data_*')

try:
    dirs_to_delete.remove('data_clean')
except ValueError:
    print('Make sure directory data_clean exists.')
    sys.exit(1)

for dir_to_delete in dirs_to_delete:
    shutil.rmtree(dir_to_delete)
