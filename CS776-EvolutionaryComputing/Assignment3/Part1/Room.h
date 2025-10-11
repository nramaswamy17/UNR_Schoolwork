#ifndef ROOM_H
#define ROOM_H

struct Room {
    double length;
    double width;
    double x;
    double y;
    
    Room() : length(0), width(0), x(0), y(0) {}
    Room(double l, double w, double px, double py) 
        : length(l), width(w), x(px), y(py) {}
    
    double area() const { return length * width; }
};

#endif // ROOM_H