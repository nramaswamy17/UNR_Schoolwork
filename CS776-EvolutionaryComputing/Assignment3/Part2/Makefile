CXX = g++
CXXFLAGS = -std=c++17 -O3 -Wall -Wextra
LDFLAGS = -no-pie -lm

TARGET = build/optimizer
SOURCES = optimizer.cpp
OBJECTS = $(SOURCES:.cpp=.o)
OBJFILE = eval_function/evalA2Bit.o

all: $(TARGET)

$(TARGET): $(OBJECTS) $(OBJFILE)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJECTS) $(OBJFILE) $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJECTS) $(TARGET)
	rm -rf data/*

test: $(TARGET)
	@echo "Running quick test..."
	./$(TARGET) ga 50 10 0.01 0.9 3 5

run: $(TARGET)
	@echo "Running GA..."
	./$(TARGET) ga 50 100 0.01 0.9 3 30
	@echo "Running HC..."
	./$(TARGET) hc 1000 30