.SILENT:

# Splits data_clean into train and test. You can change the train & test
# fractional split in split_data.py.
train: split_data.py data_clean
	python3 split_data.py

# Generates useful/cleaner form of data in "COVID-19 Radiograpy Database".
# Refer generate_raw.sh for more details.
data_clean: COVID-19\ Radiography\ Database
	bash generate_raw.sh

# Removes train, valid and test directories, retains data_clean.
almost_clean:
	rm -rf train valid test

# Removes all generated directories.
clean:
	rm -rf train valid test
	rm -rf data_clean
