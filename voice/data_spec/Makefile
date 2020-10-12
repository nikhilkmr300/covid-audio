.SILENT:

all: log_spec/Makefile mel_spec/Makefile
	cd log_spec && make
	cd mel_spec && make

log_specs: log_spec/Makefile
	cd log_spec && make

mel_specs: mel_spec/Makefile
	cd mel_spec && make

clean: log_spec/Makefile mel_spec/Makefile
	cd log_spec && make clean
	cd mel_spec && make clean
