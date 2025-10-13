# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O3 -Isrc

# Directories
SRC_DIR = src
BUILD_DIR = build
DATA_DIR = data

# Source files (only main.cpp now, rest are header-only)
SOURCES = $(SRC_DIR)/main.cpp

# Object files
OBJECTS = $(BUILD_DIR)/main.o

# Executable
TARGET = tsp_ga

# Default target
all: directories $(TARGET)

# Create necessary directories
directories:
	@mkdir -p $(BUILD_DIR)
	@mkdir -p $(DATA_DIR)

# Link
$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $^

# Compile main.cpp
$(BUILD_DIR)/main.o: main.cpp *.h
	$(CXX) $(CXXFLAGS) -c -o $@ $<

# Clean
clean:
	rm -rf $(BUILD_DIR)/* $(TARGET)

# Clean all (including data)
cleanall: clean
	rm -rf $(DATA_DIR)

# Run with default parameters
run: $(TARGET)
	./$(TARGET) burma14 42

# Run all benchmarks with different seeds
runall: $(TARGET)
	@echo "Running all benchmarks..."
	@for bench in burma14 berlin52 eil51 eil76 lin105 lin318; do \
		echo ""; \
		echo "Running $$bench..."; \
		./$(TARGET) $$bench 42; \
	done

# Help
help:
	@echo "Makefile targets:"
	@echo "  all        - Build the project (default)"
	@echo "  clean      - Remove build files"
	@echo "  cleanall   - Remove build and data files"
	@echo "  run        - Run with default parameters (burma14, seed 42)"
	@echo "  runall     - Run all benchmarks with seed 42"
	@echo "  help       - Display this help message"
	@echo ""
	@echo "Usage: ./$(TARGET) <benchmark> <seed>"
	@echo "  benchmark: burma14, berlin52, eil51, eil76, lin105, lin318"
	@echo "  seed: any integer (default: 42)"

.PHONY: all directories clean cleanall run runall help