.SILENT:

# Builds the complete system.
all:
	echo "This will take some time."
	python3 set_audio_params.py
	cd data_raw && make
	cd data_rnn && make
	cd data_spec && make
	cd data_struc && make

# Generates data in cleaner form in data_raw.
raw:
	cd data_raw && make

# Generates data as required by RNN models in data_rnn.
rnn:
	cd data_rnn && make

# Generates spectrograms as required by convolutional models in data_spec.
spec:
	cd data_spec && make

# Generates data as required by traditional ML models in structured/tabular
# format in data_struc.
struc:
	cd data_struc && make

# Cleans up data in all subdirectories.
clean:
	cd data_raw && make clean
	cd data_rnn && make clean
	cd data_spec && make clean
	cd data_struc && make clean

# Cleans up data in data_raw.
clean_raw:
	cd data_raw && make clean

# Cleans up data in data_rnn.
clean_rnn:
	cd data_rnn && make clean

# Cleans up data in data_spec.
clean_spec:
	cd data_spec && make clean

# Cleans up data in data_struc.
clean_struc:
	cd data_struc && make clean
