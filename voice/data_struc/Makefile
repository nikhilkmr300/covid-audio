.SILENT:

# Generates train-valid-test features in csv format according to the split in
# data_raw.
train: generate_features.py ../../data_raw/data_breath
	python3 generate_features.py

# Ensures that the train-valid-test audio samples exist in data_raw.
../../data_raw/data_breath:
	cd ../data_raw && make

clean:
	rm -rf data_*
