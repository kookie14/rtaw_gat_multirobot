#include "utils/utils.h"
namespace rds
{
    float round_num(float value, int n_digits)
    {
        return std::round(value * pow(10, n_digits)) / pow(10, n_digits);
    }
    // RandomGenerator::RandomGenerator(const int seed_value)
    // {
    //     _seed_value = seed_value;
    //     _generator.seed(seed_value);
    // }

    // void RandomGenerator::set_seed_value(int seed_value)
    // {
    //     _seed_value = seed_value;
    //     _generator.seed(seed_value);
    // }

    // int RandomGenerator::uniform(const int low, const int high)
    // {
    //     int_uniform_dist_.param(std::uniform_int_distribution<int>::param_type(low, high));
    //     return int_uniform_dist_(_generator);
    // }
    // float RandomGenerator::uniform(const float low, const float high)
    // {
    //     real_uniform_dist_.param(std::uniform_real_distribution<float>::param_type(low, high));
    //     return round_num(real_uniform_dist_(_generator));
    // }
    // float RandomGenerator::normal(const float mean, const float stddev)
    // {
    //     normal_uniform_dist_.param(std::normal_distribution<float>::param_type(mean, stddev));
    //     return round_num(normal_uniform_dist_(_generator));
    // }
    // std::vector<int> RandomGenerator::vector_uniform(const int low, const int high, const int num_elements)
    // {
    //     int_uniform_dist_.param(std::uniform_int_distribution<int>::param_type(low, high));
    //     std::vector<int> rand_vec;
    //     for (int i = 0; i < num_elements; i++)
    //     {
    //         rand_vec.push_back(int_uniform_dist_(_generator));
    //     }
    //     return rand_vec;
    // }
    // std::vector<float> RandomGenerator::vector_uniform(const float low, const float high, const int num_elements)
    // {
    //     real_uniform_dist_.param(std::uniform_real_distribution<float>::param_type(low, high));
    //     std::vector<float> rand_vec;
    //     for (int i = 0; i < num_elements; i++)
    //     {
    //         rand_vec.push_back(round_num(real_uniform_dist_(_generator)));
    //     }
    //     return rand_vec;
    // }
    // std::vector<float> RandomGenerator::vector_normal(const float mean, const float stddev, const int num_elements)
    // {
    //     normal_uniform_dist_.param(std::normal_distribution<float>::param_type(mean, stddev));
    //     std::vector<float> rand_vec;
    //     for (int i = 0; i < num_elements; i++)
    //     {
    //         rand_vec.push_back(round_num(normal_uniform_dist_(_generator)));
    //     }
    //     return rand_vec;
    // }

    float normalize_angle(const float angle)
    {
        return round_num(atan2(sin(angle), cos(angle)));
    }

    float calculate_difference_angle(const float angle1, const float angle2)
    {
        return normalize_angle(angle2 - angle1);
    }
    // Function for Euclidean distance calculation
    float euclidean_distance(const Point2D &p1, const Point2D &p2)
    {
        float dx = p2.x() - p1.x();
        float dy = p2.y() - p1.y();
        return round_num(hypot(dx, dy));
    }

    // Function for Manhattan distance calculation
    float manhattan_distance(const Point2D &p1, const Point2D &p2)
    {
        return round_num(fabs(p2.x() - p1.x()) + fabs(p2.y() - p1.y()));
    }

    // Function to check if two points are considered the same (based on distance)
    bool is_same_point(const Point2D &p1, const Point2D &p2)
    {
        return euclidean_distance(p1, p2) < SAME_DIST;
    }

    // Function to calculate total path cost using Euclidean distance
    float calculate_path_cost(const std::vector<Point2D> &path)
    {
        if (path.size() <= 1)
        {
            return 0.0;
        }

        float cost = 0.0;
        for (size_t i = 0; i < path.size() - 1; ++i)
        {
            cost += euclidean_distance(path[i], path[i + 1]);
        }
        return round_num(cost);
    }

    // Function to calculate angle between two points in radians
    float angle_by_two_point(const Point2D &p1, const Point2D &p2)
    {
        return round_num(std::atan2(p2.y() - p1.y(), p2.x() - p1.x()));
    }

    float dot_product(const Point2D &v1, const Point2D &v2)
    {
        return v1.x() * v2.x() + v1.y() * v2.y();
    }
    float magnitude(const Point2D &v1)
    {
        return hypot(v1.x(), v1.y());
    }

    float angle_by_two_vectors(const Point2D &A, const Point2D &B, const Point2D &C)
    {
        Point2D v1 = B - A;
        Point2D v2 = C - A;
        return acos(dot_product(v1, v2) / (magnitude(v1) * magnitude(v2) + 1e-10));
    }

    bool is_collinear(const Point2D &p1, const Point2D &p2, const Point2D &p3)
    {
        // Calculate slopes
        float slope1 = (p2.y() - p1.y()) * (p3.x() - p2.x());
        float slope2 = (p3.y() - p2.y()) * (p2.x() - p1.x());

        // Check if slopes are equal (or close enough)
        return fabs(slope1 - slope2) < 1e-10;
    }

    // Calculate distance between a point && a line
    float point_line_distance(const Point2D &start, const Point2D &end, const Point2D &point)
    {
        float px = end.x() - start.x();
        float py = end.y() - start.y();

        float norm = px * px + py * py;
        if (norm == 0.0)
        {
            // Handle case where start && end are the same point
            return euclidean_distance(start, point);
        }

        float u = ((point.x() - start.x()) * px + (point.y() - start.y()) * py) / norm;

        if (u > 1.0)
        {
            u = 1.0;
        }
        else if (u < 0.0)
        {
            u = 0.0;
        }

        float x = start.x() + u * px;
        float y = start.y() + u * py;

        float dx = x - point.x();
        float dy = y - point.y();

        return hypot(dx, dy);
    }

    // Check if a point C is within a segment AB
    bool point_is_in_segment(const Point2D &A, const Point2D &B, const Point2D &C)
    {
        if (is_same_point(C, B))
            return true;
        if (is_same_point(A, C))
            return false;
        // if (is_same_point(C, A)) return true;
        float dist1 = euclidean_distance(A, C);
        float dist2 = euclidean_distance(B, C);
        float dist3 = euclidean_distance(A, B);
        return (dist1 + dist2 <= dist3 + SAME_DIST && dist1 + dist2 >= dist3 - SAME_DIST);
    }
    Point2D find_intersection_point(const Point2D &line1_start, const Point2D &line1_end,
                                    const Point2D &line2_start, const Point2D &line2_end)
    {
        // Calculate the intersection point
        float denominator = (line1_start.x() - line1_end.x()) * (line2_start.y() - line2_end.y()) - (line1_start.y() - line1_end.y()) * (line2_start.x() - line2_end.x());
        if (denominator == 0.0)
        {
            return Point2D(); // Lines are parallel
        }
        else
        {
            float intersect_x = round_num(((line1_start.x() * line1_end.y() - line1_start.y() * line1_end.x()) * (line2_start.x() - line2_end.x()) - (line1_start.x() - line1_end.x()) * (line2_start.x() * line2_end.y() - line2_start.y() * line2_end.x())) / denominator);
            float intersect_y = round_num(((line1_start.x() * line1_end.y() - line1_start.y() * line1_end.x()) * (line2_start.y() - line2_end.y()) - (line1_start.y() - line1_end.y()) * (line2_start.x() * line2_end.y() - line2_start.y() * line2_end.x())) / denominator);
            return Point2D(intersect_x, intersect_y);
        }
    }

    // Function to check segment intersection
    std::pair<bool, Point2D> check_segment_intersection(const Point2D &line1_start, const Point2D &line1_end,
                                                        const Point2D &line2_start, const Point2D &line2_end)
    {
        // Calculate vectors representing the lines
        Point2D line1 = line1_end - line1_start;
        Point2D line2 = line2_end - line2_start;

        // Check for special cases (coincident or overlapping segments)
        if (is_same_point(line1_end, line2_end) && !is_same_point(line1_start, line2_start))
            return std::make_pair(true, line1_end);
        if (is_same_point(line1_end, line2_start) && !is_same_point(line1_start, line2_end))
            return std::make_pair(true, line1_end);
        if (is_same_point(line1_start, line2_start) && !is_same_point(line1_end, line2_end))
            return std::make_pair(true, line1_start);
        if (is_same_point(line1_start, line2_end) && !is_same_point(line1_end, line2_start))
            return std::make_pair(true, line1_start);
        // Check for collinear (lines may overlap partially)
        if (is_collinear(line1_start, line1_end, line2_start) && point_is_in_segment(line1_start, line1_end, line2_start))
            return std::make_pair(true, line2_start);
        if (is_collinear(line1_start, line1_end, line2_end) && point_is_in_segment(line1_start, line1_end, line2_end))
            return std::make_pair(true, line2_end);
        if (is_collinear(line2_start, line2_end, line1_start) && point_is_in_segment(line2_start, line2_end, line1_start))
            return std::make_pair(true, line1_start);
        if (is_collinear(line2_start, line2_end, line1_end) && point_is_in_segment(line2_start, line2_end, line1_end))
            return std::make_pair(true, line1_end);

        // Calculate denominator for intersection check
        float denom = line1.x() * line2.y() - line2.x() * line1.y();
        if (fabs(denom) < 1e-10)
        {
            // Lines are parallel
            return std::make_pair(false, Point2D());
        }

        bool denom_positive = denom > 0;

        // Calculate auxiliary vector
        Point2D aux = line1_start - line2_start;

        // Numerators for intersection parameters
        float s_numer = line1.x() * aux.y() - line1.y() * aux.x();
        float t_numer = line2.x() * aux.y() - line2.y() * aux.x();

        // Check if intersection falls within segments using signs
        if ((s_numer < 0) == denom_positive)
            return std::make_pair(false, Point2D());
        if ((t_numer < 0) == denom_positive)
            return std::make_pair(false, Point2D());
        if (((s_numer > denom) == denom_positive) || ((t_numer > denom) == denom_positive))
            return std::make_pair(false, Point2D());

        // Calculate intersection point
        float t = t_numer / denom;
        return std::make_pair(true, Point2D(line1_start.x() + t * line1.x(), line1_start.y() + t * line1.y()));
    }

    std::vector<std::vector<float>> calculate_rectangle_coordinate(float center_x, float center_y, float angle,
                                                                   float length, float width)
    {
        float halfLength = length / 2.0;
        float halfWidth = width / 2.0;
        float sinAngle = std::sin(angle);
        float cosAngle = std::cos(angle);

        // Bottom left
        std::vector<float> x_shape, y_shape;
        x_shape.push_back(round_num(center_x + (cosAngle * -halfLength) - (sinAngle * halfWidth)));
        y_shape.push_back(round_num(center_y + (sinAngle * -halfLength) + (cosAngle * halfWidth)));
        // Top left corner
        x_shape.push_back(round_num(center_x + (cosAngle * -halfLength) - (sinAngle * -halfWidth)));
        y_shape.push_back(round_num(center_y + (sinAngle * -halfLength) + (cosAngle * -halfWidth)));
        // Top right
        x_shape.push_back(round_num(center_x + (cosAngle * halfLength) - (sinAngle * -halfWidth)));
        y_shape.push_back(round_num(center_y + (sinAngle * halfLength) + (cosAngle * -halfWidth)));
        // Bottom right
        x_shape.push_back(round_num(center_x + (cosAngle * halfLength) - (sinAngle * halfWidth)));
        y_shape.push_back(round_num(center_y + (sinAngle * halfLength) + (cosAngle * halfWidth)));
        // Bottom left
        x_shape.push_back(round_num(center_x + (cosAngle * -halfLength) - (sinAngle * halfWidth)));
        y_shape.push_back(round_num(center_y + (sinAngle * -halfLength) + (cosAngle * halfWidth)));
        return {x_shape, y_shape};
    }

    std::vector<std::vector<float>> load_txt(const std::string &path)
    {
        std::ifstream file(path);
        if (!file.is_open())
        {
            throw std::runtime_error("Could not open file");
        }

        std::vector<std::vector<float>> data;
        std::string line;

        while (std::getline(file, line))
        {
            std::stringstream ss(line);
            std::vector<float> row;
            float value;
            while (ss >> value)
            {
                row.push_back(value);
            }
            data.push_back(row);
        }

        file.close();

        if (data.empty())
        {
            throw std::runtime_error("No data found in file");
        }

        return data;
    }
    matplot::vector_1d vector_to_plot_1d(const std::vector<float> &v)
    {
        matplot::vector_1d out;
        for (int i = 0; i < v.size(); i++)
            out.push_back(v[i]);
        return out;
    }
    matplot::vector_2d vector_to_plot_2d(const std::vector<std::vector<float>> &v)
    {
        matplot::vector_2d out2d;
        for (int i = 0; i < v.size(); i++)
        {
            matplot::vector_1d out1d;
            for (int j = 0; j < v[i].size(); j++)
            {
                out1d.push_back(v[i][j]);
            }
            out2d.push_back(out1d);
        }
        return out2d;
    }

    matplot::vector_2d point2d_to_plot_2d(const std::vector<Point2D> &v)
    {
        matplot::vector_2d out2d;
        out2d.push_back({});
        out2d.push_back({});
        for (int i = 0; i < v.size(); i++)
        {
            out2d[0].push_back(v[i].x());
            out2d[1].push_back(v[i].y());
        }
        return out2d;
    }
}