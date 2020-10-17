.SILENT:

# Generates train-valid-test features in csv format according to the split in
# data_raw.
train: generate_features.py
	cd ../data_raw && make
	python3 generate_features.py

# Removes generated directories.
clean:
	rm -rf data_*
