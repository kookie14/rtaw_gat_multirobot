#pragma once
#ifndef UTILS_H_
#define UTILS_H_
#include <iostream>
#include <cmath>
#include <random>
#include <vector>
#include <fstream>
#include <sstream>
#include <matplot/matplot.h>
#include <assert.h>
#include <torch/torch.h>
const int N_DIGITS = 4;
// Distance between same point
const float SAME_DIST = 0.02;
// #include <torch/torch.h>
namespace rds
{
    float round_num(float value, int n_digits = N_DIGITS);
    // class RandomGenerator
    // {
    // private:
    //     std::mt19937 _generator; // Mersenne Twister generator
    //     int _seed_value; // Seed value
    //     std::uniform_int_distribution<int> int_uniform_dist_;
    //     std::uniform_real_distribution<float> real_uniform_dist_;
    //     std::normal_distribution<float> normal_uniform_dist_;
    // public:
    //     RandomGenerator(const int seed_value = 10);
    //     ~RandomGenerator(){}
    //     void set_seed_value(int seed_value);
    //     int uniform(const int low, const int high);
    //     float uniform(const float low, const float high);
    //     float normal(const float mean, const float stddev);
    //     std::vector<int> vector_uniform(const int low, const int high, const int num_elements= 2);
    //     std::vector<float> vector_uniform(const float low, const float high, const int num_elements = 2);
    //     std::vector<float> vector_normal(const float mean, const float stddev, const int num_elements = 2);
    // };

    float normalize_angle(const float angle);
    float calculate_difference_angle(const float angle1, const float angle2);

    class Point2D
    {
    private:
        float x_;
        float y_;

    public:
        // Constructor with default arguments
        Point2D(float x = 0.0, float y = 0.0) : x_{round_num(x)}, y_{round_num(y)} {}
        Point2D(std::vector<float> point) : x_{round_num(point[0])}, y_{round_num(point[1])} {}
        // Getters for x and y coordinates
        float x() const { return x_; }
        float y() const { return y_; }

        // Setters for x and y coordinates
        void set_x(float x) { x_ = round_num(x); }
        void set_y(float y) { y_ = round_num(y); }

        // Set both x and y coordinates
        void set(float x, float y)
        {
            x_ = round_num(x);
            y_ = round_num(y);
        }

        // Convert to a std::vector
        std::vector<float> to_vector() const
        {
            return std::vector<float>{x_, y_};
        }

        // Overload arithmetic operators (+, -, *, /) for Point2D objects
        Point2D operator+(const Point2D &other) const
        {
            return Point2D(x_ + other.x_, y_ + other.y_);
        }

        Point2D operator-(const Point2D &other) const
        {
            return Point2D(x_ - other.x_, y_ - other.y_);
        }

        Point2D operator*(const float &scalar) const
        {
            return Point2D(x_ * scalar, y_ * scalar);
        }

        Point2D operator/(const float &scalar) const
        {
            assert(fabs(scalar) > 1e-10);
            return Point2D(x_ / scalar, y_ / scalar);
        }

        // Overload string stream insertion operator for printing
        friend std::ostream &operator<<(std::ostream &os, const Point2D &point)
        {
            os << "x = " << point.x_ << ", y = " << point.y_;
            return os;
        }
    };

    class Velocity
    {
    private:
        float linear_;
        float omega_;

    public:
        // Constructor with default arguments
        Velocity(float linear = 0.0, float omega = 0.0) : linear_{round_num(linear)}, omega_{round_num(omega)} {}
        Velocity(std::vector<float> vel) : linear_{round_num(vel[0])}, omega_{round_num(vel[1])} {}
        // Getters for linear and omega
        float linear() const { return linear_; }
        float omega() const { return omega_; }

        // Setters for linear and omega
        void set_linear(float linear) { linear_ = round_num(linear); }
        void set_y(float omega) { omega_ = round_num(omega); }

        // Set both linear and omega
        void set(float linear, float omega)
        {
            linear_ = round_num(linear);
            omega_ = round_num(omega);
        }

        // Convert to a std::vector
        std::vector<float> to_vector() const
        {
            return std::vector<float>{linear_, omega_};
        }

        // Overload arithmetic operators (+, -, *, /) for Velocity objects
        Velocity operator+(const Velocity &other) const
        {
            return Velocity(linear_ + other.linear_, omega_ + other.omega_);
        }

        Velocity operator-(const Velocity &other) const
        {
            return Velocity(linear_ - other.linear_, omega_ - other.omega_);
        }

        Velocity operator*(const float &scalar) const
        {
            return Velocity(linear_ * scalar, omega_ * scalar);
        }

        Velocity operator/(const float &scalar) const
        {
            assert(fabs(scalar) > 1e-10);
            return Velocity(linear_ / scalar, omega_ / scalar);
        }

        // Overload string stream insertion operator for printing
        friend std::ostream &operator<<(std::ostream &os, const Velocity &point)
        {
            os << "linear = " << point.linear_ << ", y = " << point.omega_;
            return os;
        }
    };

    class Pose2D
    {
    private:
        float x_;
        float y_;
        float theta_;

    public:
        // Constructor with default arguments
        Pose2D(float x = 0.0, float y = 0.0, float theta = 0.0) : x_(x), y_(y), theta_(normalize_angle(theta)) {}

        // Getters for x, y, and theta
        float x() const { return x_; }
        float y() const { return y_; }
        float theta() const { return theta_; }

        // Get position as a Point2D object
        Point2D position() const { return Point2D(x_, y_); }

        // Setters for x, y, and theta
        void set_x(float x) { x_ = round_num(x); }
        void set_y(float y) { y_ = round_num(y); }
        void set_theta(float theta) { theta_ = normalize_angle(theta); }

        // Setters for x, y, and theta together
        void set(float x, float y, float theta)
        {
            x_ = round_num(x);
            y_ = round_num(y);
            theta_ = normalize_angle(theta);
        }

        // Convert to a std::vector
        std::vector<float> toVector() const
        {
            return std::vector<float>{x_, y_, theta_};
        }

        // Overload arithmetic operators (+, -) for Pose2D objects
        Pose2D operator+(const Pose2D &other) const
        {
            return Pose2D(x_ + other.x_, y_ + other.y_, theta_ + other.theta_);
        }

        Pose2D operator-(const Pose2D &other) const
        {
            return Pose2D(x_ - other.x_, y_ - other.y_, normalize_angle(theta_ - other.theta_));
        }

        // Overload string stream insertion operator for printing
        friend std::ostream &operator<<(std::ostream &os, const Pose2D &pose)
        {
            os << "x = " << pose.x_ << ", y = " << pose.y_ << ", theta = " << round_num(pose.theta_ * 180 * M_1_PI);
            return os;
        }
    };

    float euclidean_distance(const Point2D &p1, const Point2D &p2);
    float manhattan_distance(const Point2D &p1, const Point2D &p2);
    bool is_same_point(const Point2D &p1, const Point2D &p2);
    float calculate_path_cost(const std::vector<Point2D> &path);
    float angle_by_two_point(const Point2D &p1, const Point2D &p2);
    float dot_product(const Point2D &v1, const Point2D &v2);
    float magnitude(const Point2D &v1);
    float angle_by_two_vectors(const Point2D &A, const Point2D &B, const Point2D &C);
    bool is_collinear(const Point2D &p1, const Point2D &p2, const Point2D &p3);
    float point_line_distance(const Point2D &start, const Point2D &end, const Point2D &point);
    bool point_is_in_segment(const Point2D &start, const Point2D &end, const Point2D &p);
    Point2D find_intersection_point(const Point2D &line1_start, const Point2D &line1_end,
                                    const Point2D &line2_start, const Point2D &line2_end);
    std::pair<bool, Point2D> check_segment_intersection(const Point2D &line1_start, const Point2D &line1_end,
                                                        const Point2D &line2_start, const Point2D &line2_end);
    std::vector<std::vector<float>> calculate_rectangle_coordinate(float center_x, float center_y, float angle,
                                                                   float length, float width);
    std::vector<std::vector<float>> load_txt(const std::string &path);
    matplot::vector_1d vector_to_plot_1d(const std::vector<float> &v);
    matplot::vector_2d vector_to_plot_2d(const std::vector<std::vector<float>> &v);
    matplot::vector_2d point2d_to_plot_2d(const std::vector<Point2D> &v);

    template <typename T>
    bool vector_exists(const std::vector<std::vector<T>> &container, const std::vector<T> &target)
    {
        auto it = std::find(container.begin(), container.end(), target);
        return it != container.end();
    }
    template <typename T>
    bool number_exists(const std::vector<T> &container, const T &value)
    {
        auto it = std::find(container.begin(), container.end(), value);
        return it != container.end();
    }
}
#endif // UTILS_H_