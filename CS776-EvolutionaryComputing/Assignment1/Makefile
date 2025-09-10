CXX = g++
CXXFLAGS = -std=c++20 -Wall -Wextra

# Default target - run all experiments
all: clean blackbox1 blackbox2 easy hard

# Clean data directory
clean:
	rm -rf data/*

# Black box hill climber executables
bb1: black_box_hill_climber.cpp black_box/eval.o
	$(CXX) $(CXXFLAGS) -o build/bb1 black_box_hill_climber.cpp black_box/eval.o

bb2: black_box_hill_climber.cpp black_box/eval1.o
	$(CXX) $(CXXFLAGS) -o build/bb2 black_box_hill_climber.cpp black_box/eval1.o

# Black box experiment 1
blackbox1: bb1
	./build/bb1 bb1

# Black box experiment 2  
blackbox2: bb2
	./build/bb2 bb2

# Easy function (onemax)
easy: assignment1.cpp ../Assignment0/dejong_functions.cpp
	$(CXX) $(CXXFLAGS) -o build/assignment1 assignment1.cpp ../Assignment0/dejong_functions.cpp
	./build/assignment1

# Hard function (deceptive trap)
hard: assignment1.cpp ../Assignment0/dejong_functions.cpp
	$(CXX) $(CXXFLAGS) -o build/assignment1 assignment1.cpp ../Assignment0/dejong_functions.cpp
	./build/assignment1

# Original target for backward compatibility
assignment1: easy