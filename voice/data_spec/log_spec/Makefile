.SILENT:

# Generates log spectrograms of audio samples in data_raw, organized by audio
# type, and using the train-valid-test split in each data_raw_<audio_type>.
spec_breath: generate_log_specs.py
	cd ../../data_raw && make
	python3 generate_log_specs.py

# Deletes the directories containing spectrograms. Run with care, the
# spectrograms take a while to be generated.
clean:
	rm -rf spec_*
