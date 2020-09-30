.SILENT:

# Splits data_clean into train, valid and test for each audio type. You can
# change the train:valid:test split ratio in split_data.py.
data_breath: split_data.py data_clean
	python3 split_data.py

# Generates useful/cleaner form of data in "KDD_paper_data".
# Refer generate_data_clean.sh for more details.
data_clean: generate_data_clean.sh KDD_paper_data
	bash generate_data_clean.sh

# Removes directories created for each class of audio sample, retains
# data_clean.
almost_clean: delete_audio_type_dirs.py
	python3 delete_audio_type_dirs.py

# Removes all generated directories.
clean:
	rm -r data_*
