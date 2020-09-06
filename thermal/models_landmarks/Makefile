.SILENT:

# Makes the annot directory containing XML annotation files for the landmarks.
annot: ../train
	python3 generate_xml.py

# Ensures that train, valid and test directories have been created in ../data
../train:
	cd ../data && make

clean:
	rm -rf annot
