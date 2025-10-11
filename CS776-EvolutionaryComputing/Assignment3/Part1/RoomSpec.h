#ifndef ROOMSPEC_H
#define ROOMSPEC_H

#include <string>

struct RoomSpec {
    std::string name;
    double minLength, maxLength;
    double minWidth, maxWidth;
    double minArea, maxArea;
    double aspectRatio;  // 0 if no constraint
    double costMultiplier;
    
    RoomSpec(std::string n, double minL, double maxL, double minW, double maxW,
             double minA, double maxA, double ratio, double cost)
        : name(n), minLength(minL), maxLength(maxL), minWidth(minW), 
          maxWidth(maxW), minArea(minA), maxArea(maxA), 
          aspectRatio(ratio), costMultiplier(cost) {}
};

#endif // ROOMSPEC_H