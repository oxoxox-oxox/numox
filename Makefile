CC = gcc
CFLAGS = -g -Wall -std=c99 -I. -Isrc/c -fopenmp -mavx2 -mfma -pthread
LDFLAGS = -fopenmp
CUNIT = -lcunit
PYTHON = $(shell python3-config --cflags) $(shell python3-config --ldflags --embed)

install:
	rm -rf build
	python3 -m pip install --no-build-isolation .

uninstall:
	python3 -m pip uninstall -y numox

clean:
	rm -f *.o test test.exe
	rm -rf build dist *.egg-info src/*.egg-info __pycache__
	rm -f src/numox/*.so src/numox/*.pyd

test:
	rm -f test
	$(CC) $(CFLAGS) ./tests/mat_test.c ./src/c/matrix.c -o test $(LDFLAGS) $(CUNIT) $(PYTHON)
	./test

.PHONY: install uninstall clean test
