.SILENT:

# Generates Mel spectrograms of audio samples in data_raw, organized by audio
# type, and using the train-valid-test split in each data_raw_<audio_type>.
spec_breath: generate_mel_specs.py ../../data_raw/data_breath
	python3 generate_mel_specs.py

# Ensures that the train-valid-test audio samples exist in data_raw.
../../data_raw/data_breath:
	cd ../../data_raw && make

# Deletes the directories containing spectrograms. Run with care, the
# spectrograms take a while to be generated.
clean:
	rm -rf spec_*
