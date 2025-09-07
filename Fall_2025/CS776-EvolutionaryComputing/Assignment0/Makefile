CXX = g++
CXXFLAGS = -std=c++20 -Wall -Wextra

# Main program
assignment0: assignment0.cpp dejong_functions.cpp
	rm -rf data/*
	$(CXX) $(CXXFLAGS) -o build/assignment0 assignment0.cpp dejong_functions.cpp
	./build/assignment0

# Test program
test: test_assignment0.cpp dejong_functions.cpp
	$(CXX) $(CXXFLAGS) -o build/test_assignment0 test_assignment0.cpp dejong_functions.cpp
	./build/test_assignment0

# Clean up
clean:
	rm -f build/assignment0 build/test_assignment0

# Default target
all: assignment0 test

.PHONY: clean test run-tests all
