.SILENT:

# Ensures that train, valid and test directories have been created in ../data.
../data/train:
	cd ../data && make

# Generates results of CNN model (cnn_predict.py) and writes images to
# ../results_face/results_cnn.
cnn: ../data/train
	python3 cnn_predict.py

# Generates results of HOG+SVM model (hog_predict.py) and writes images to
# ../results_face/results_hog.
hog: ../data/train
	python3 hog_predict.py
