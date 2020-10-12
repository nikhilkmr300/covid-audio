.SILENT:

all: data/Makefile models_face/Makefile models_landmarks/Makefile
	cd data && make
	cd models_face && make cnn && make hog
	cd models_landmarks && make

clean: data/Makefile models_face/Makefile models_landmarks/Makefile
	cd data && make clean
	# cd models_face && make clean
	cd models_landmarks && make clean
