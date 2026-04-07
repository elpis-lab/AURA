#ifndef SE2_H
#define SE2_H

#include <vector>
#include <cmath>

// Simple SE2 Pose class for transformations
class SE2Pose {
public:
    double x, y, theta;
    
    SE2Pose(double x = 0.0, double y = 0.0, double theta = 0.0) 
        : x(x), y(y), theta(theta) {}
    
    // Create from vector [x, y, theta]
    SE2Pose(const std::vector<double>& pose) 
        : x(pose[0]), y(pose[1]), theta(pose[2]) {}
    
    // Composition operator: this_pose @ other_pose
    SE2Pose operator*(const SE2Pose& other) const {
        double cos_t = cos(theta);
        double sin_t = sin(theta);
        
        return SE2Pose(
            x + cos_t * other.x - sin_t * other.y,
            y + sin_t * other.x + cos_t * other.y,
            theta + other.theta
        );
    }
    
    // Inverse pose
    SE2Pose inverse() const {
        double cos_t = cos(-theta);
        double sin_t = sin(-theta);
        
        return SE2Pose(
            cos_t * (-x) - sin_t * (-y),
            sin_t * (-x) + cos_t * (-y),
            -theta
        );
    }
    
    // Difference: other_pose - this_pose (in local frame)
    SE2Pose difference(const SE2Pose& other) const {
        return inverse() * other;
    }
    
    // To vector
    std::vector<double> toVector() const {
        return {x, y, theta};
    }
};

#endif // SE2_H 