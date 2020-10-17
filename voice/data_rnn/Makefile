.SILENT:

# Generates train-valid-test features in csv format according to the split in
# data_raw.
train:
	cd ../data_raw && make
	python3 generate_data_rnn.py

# Removes generated directories.
clean:
	rm -rf data_*
