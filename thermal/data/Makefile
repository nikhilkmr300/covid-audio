.SILENT:

# Generates directories train, valid and test.
train: split_data.py FaceDB_Snapshot_complete
	echo "Splitting data into training, validation and test sets..."
	rm -rf train valid test
	# Default split is 0.8, 0, 0.2. You can change it in split_data.py.
	python3 split_data.py

# Removes all generated files and directories.
clean:
	rm -rf train valid test
