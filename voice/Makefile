.SILENT:

# Builds the complete system.
all: data_raw/Makefile data_rnn/Makefile data_spec/Makefile data_struc/Makefile
	echo "This will take some time."
	python3 set_audio_params.py
	cd data_raw && make
	cd data_rnn && make
	cd data_spec && make
	cd data_struc && make

# Generates data in cleaner form in data_raw.
data_raw: data_raw/Makefile
	cd data_raw && make

# Generates data as required by RNN models in data_rnn.
data_rnn: data_rnn/Makefile
	cd data_rnn && make

# Generates spectrograms as required by convolutional models in data_spec.
data_spec: data_spec/Makefile
	cd data_spec && make

# Generates data as required by traditional ML models in structured/tabular
# format in data_struc.
data_struc: data_struc/Makefile
	cd data_struc && make

clean: data_raw/Makefile data_rnn/Makefile data_spec/Makefile data_struc/Makefile
	cd data_raw && make clean
	cd data_rnn && make clean
	cd data_spec && make clean
	cd data_struc && make clean
