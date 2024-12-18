#pragma once
#ifndef MAP_AND_GRAPH_H_
#define MAP_AND_GRAPH_H_
#include "utils/utils.h"
#include "utils/type_and_state.h"
#include <unordered_map>
#include <queue>
#include <unordered_set>
#include <algorithm>
namespace rds
{
    class Line
    {
    private:
        int type_;
        Point2D center_;
        float length_;
        float width_;
        std::vector<std::vector<float>> _coords;

    public:
        Line(int type_ = NONE_LINE, const Point2D &center = Point2D(), float length = 0.0f, float width = 0.0f)
            : type_(type_), center_(center), length_(round_num(length)), width_(round_num(width))
        {
            if (type_ == DOUBLE_HORIZONTAL || type_ == SINGLE_HORIZONTAL)
            {
                _coords = calculate_rectangle_coordinate(center_.x(), center_.y(), 0.0, length_, width_);
            }
            else
            {
                _coords = calculate_rectangle_coordinate(center_.x(), center_.y(), M_PI / 2, length_, width_);
            }
        }

        float x_center() const { return center_.x(); }
        float y_center() const { return center_.y(); }
        Point2D center() const { return center_; }
        std::vector<std::vector<float>> coords() const { return _coords; }
        float length() const { return length_; }
        float width() const { return width_; }
        int type() const { return type_; }
        friend std::ostream &operator<<(std::ostream &os, const Line &line)
        {
            std::string type_str;
            switch (line.type_)
            {
            case DOUBLE_HORIZONTAL:
                type_str = "double horizontal";
                break;
            case SINGLE_HORIZONTAL:
                type_str = "single horizontal";
                break;
            case DOUBLE_VERTICAL:
                type_str = "double vertical";
                break;
            case SINGLE_VERTICAL:
                type_str = "single vertical";
                break;
            default:
                type_str = "unknown";
                break;
            }

            std::string line_str = "Type: " + type_str + " | Center: [" + std::to_string(line.center_.x()) + ", " +
                                   std::to_string(line.center_.y()) + "] | Length: " + std::to_string(line.length_) +
                                   " | Width: " + std::to_string(line.width_);
            os << line_str;
            return os;
        }
    };

    class Zone
    {
    private:
        int type_;
        Point2D center_;
        float length_;
        float width_;
        std::vector<std::vector<float>> coords_;

    public:
        Zone(int type_ = NONE_ZONE, const Point2D &center = Point2D(), float length = 0.0, float width = 0.0)
            : type_(type_), center_(center), length_(round_num(length)), width_(round_num(width))
        {
            coords_ = calculate_rectangle_coordinate(center_.x(), center_.y(), 0.0, length_, width_);
        }

        int type() const { return type_; }
        float x_center() const { return center_.x(); }
        float y_center() const { return center_.y(); }
        Point2D center() const { return center_; }
        std::vector<std::vector<float>> coords() const { return coords_; }
        float length() const { return length_; }
        float width() const { return width_; }

        friend std::ostream &operator<<(std::ostream &os, const Zone &zone)
        {
            std::string type_str;
            switch (zone.type_)
            {
            case WORKING_ZONE:
                type_str = "working";
                break;
            case CHARGING_ZONE:
                type_str = "charging";
                break;
            case STORAGE_ZONE:
                type_str = "storage";
                break;
            case WAITING_ZONE:
                type_str = "waiting";
                break;
            default:
                type_str = "unknown";
            }

            std::string zone_str = "Type: " + type_str + " | Center: [" + std::to_string(zone.center_.x()) + ", " +
                                   std::to_string(zone.center_.y()) + "] | Length: " + std::to_string(zone.length_) +
                                   " | Width: " + std::to_string(zone.width_);
            return os << zone_str;
        }
    };

    class Point
    {
    private:
        int point_type_;
        int line_type_;
        Point2D center_;
        float length_;
        float width_;
        std::vector<std::vector<float>> coords_;

    public:
        Point(int point_type, int line_type, const Point2D &center, float length, float width)
            : point_type_(point_type), line_type_(line_type), center_(center), length_(length), width_(width)
        {
            if (line_type == DOUBLE_HORIZONTAL || line_type == SINGLE_HORIZONTAL)
            {
                coords_ = calculate_rectangle_coordinate(center_.x(), center_.y(), M_PI / 2, length_, width_);
            }
            else
            {
                coords_ = calculate_rectangle_coordinate(center_.x(), center_.y(), 0.0, length_, width_);
            }
        }

        int point_type() const { return point_type_; }
        int line_type() const { return line_type_; }
        float x() const { return center_.x(); }
        float y() const { return center_.y(); }
        Point2D center() const { return center_; }
        std::vector<std::vector<float>> coords() const { return coords_; }
        float length() const { return length_; }
        float width() const { return width_; }

        friend std::ostream &operator<<(std::ostream &os, const Point &p)
        {
            std::string point_type_str;
            switch (p.point_type_)
            {
            case WORKING_VERTEX:
                point_type_str = "working";
                break;
            case CHARGING_VERTEX:
                point_type_str = "charging";
                break;
            case STORAGE_VERTEX:
                point_type_str = "storage";
                break;
            case WAITING_VERTEX:
                point_type_str = "waiting";
                break;
            default:
                point_type_str = "unknown";
                break;
            }

            std::string line_type_str;
            switch (p.line_type_)
            {
            case DOUBLE_HORIZONTAL:
                line_type_str = "double horizontal";
                break;
            case SINGLE_HORIZONTAL:
                line_type_str = "single horizontal";
                break;
            case DOUBLE_VERTICAL:
                line_type_str = "double vertical";
                break;
            case SINGLE_VERTICAL:
                line_type_str = "single vertical";
                break;
            default:
                line_type_str = "unknown";
                break;
            }

            std::string point_str = "Point type: " + point_type_str + " | Line type: " + line_type_str +
                                    " | Center: [" + std::to_string(p.center_.x()) + ", " + std::to_string(p.center_.y()) +
                                    "] | Length: " + std::to_string(p.length_) + " | Width: " + std::to_string(p.width_);
            return os << point_str;
        }
    };

    class Vertex
    {
    private:
        int id_;
        int type_;
        Point2D position_;
        int h_direct_;
        int v_direct_;
        int line_type_;
        std::vector<int> neighbors_;
        std::unordered_map<int, int> edge_ids_;

    public:
        Vertex(int id = -1, int type = UNKNOWN_VERTEX, const Point2D &position = Point2D(), int h_direct = UNDIRECTED,
               int v_direct = UNDIRECTED, int line_type = NONE_LINE, const std::vector<int> &neighbors = {},
               const std::unordered_map<int, int> &edge_ids = {})
            : id_(id), type_(type), position_(position), h_direct_(h_direct), v_direct_(v_direct),
              line_type_(line_type), neighbors_(neighbors), edge_ids_(edge_ids)
        {
        }

        int id() const { return id_; }
        float x() const { return position_.x(); }
        float y() const { return position_.y(); }
        Point2D position() const { return position_; }
        std::vector<int> neighbors() const { return neighbors_; }
        int edge_id(int neighbor_id) const { return edge_ids_.at(neighbor_id); }

        void add_neighbor(int neighbor_id, int edge_id)
        {
            neighbors_.push_back(neighbor_id);
            edge_ids_[neighbor_id] = edge_id;
        }

        int line_type() const { return line_type_; }
        int type() const { return type_; }
        int h_direct() const { return h_direct_; }
        int v_direct() const { return v_direct_; }
        friend std::ostream &operator<<(std::ostream &os, const Vertex &v)
        {
            std::string type_str;
            switch (v.type_)
            {
            case WORKING_VERTEX:
                type_str = "working";
                break;
            case CHARGING_VERTEX:
                type_str = "charging";
                break;
            case STORAGE_VERTEX:
                type_str = "storage";
                break;
            case WAITING_VERTEX:
                type_str = "waiting";
                break;
            case ROBOT_VERTEX:
                type_str = "robot";
                break;
            case BOUNDARY_VERTEX:
                type_str = "boundary";
                break;
            case LINE_VERTEX:
                type_str = "line";
                break;
            default:
                type_str = "unknown";
                break;
            }

            std::string line_type_str;
            switch (v.line_type_)
            {
            case DOUBLE_HORIZONTAL:
                line_type_str = "double horizontal";
                break;
            case SINGLE_HORIZONTAL:
                line_type_str = "single horizontal";
                break;
            case DOUBLE_VERTICAL:
                line_type_str = "double vertical";
                break;
            case SINGLE_VERTICAL:
                line_type_str = "single vertical";
                break;
            default:
                line_type_str = "unknown";
                break;
            }

            std::string vertex_str = "ID: " + std::to_string(v.id_) + " | Vertex type: " + type_str +
                                     " | Line type: " + line_type_str + " | Position: [" +
                                     std::to_string(v.position_.x()) + ", " + std::to_string(v.position_.y()) +
                                     "] | Neighbors: [";
            for (int i = 0; i < v.neighbors_.size(); i++)
            {
                if (i < v.neighbors_.size() - 1)
                    vertex_str += std::to_string(v.neighbors_[i]) + ", ";
                else
                    vertex_str += std::to_string(v.neighbors_[i]);
            }
            vertex_str += "]";
            return os << vertex_str;
        }
    };

    class Edge
    {
    private:
        int id_;
        int start_id_;
        int end_id_;
        float distance_;
        float direction_;
        float velocity_;

    public:
        Edge(int id = -1, const Vertex &start = Vertex(), const Vertex &end = Vertex(),
             const float velocity = 1.0f, const float distance = 0.0f, const float direction = 0.0f)
            : id_(id), start_id_(start.id()), end_id_(end.id()), velocity_(velocity)
        {
            if (distance <= 0.01)
            {
                distance_ = euclidean_distance(start.position(), end.position());
            }
            else
            {
                distance_ = distance;
            }
            if (direction == 0.0f)
            {
                direction_ = angle_by_two_point(start.position(), end.position());
            }
            else
            {
                direction_ = direction;
            }
        }

        int id() const { return id_; }
        void set_id(int id)
        {
            id_ = id;
        }
        int start_id() const { return start_id_; }
        int end_id() const { return end_id_; }
        float distance() const { return distance_; }
        float direction() const { return direction_; }
        float velocity() const { return velocity_; }
        friend std::ostream &operator<<(std::ostream &os, const Edge &e)
        {
            return os << "ID: " + std::to_string(e.id_) + " | Start ID: " + std::to_string(e.start_id_) +
                             " | End ID: " + std::to_string(e.end_id_) + " | Distance: " + std::to_string(e.distance_) +
                             " | Direction: " + std::to_string(e.direction_) + " | Velocity: " + std::to_string(e.velocity_);
        }
    };

    class Graph
    {
    private:
        std::vector<Vertex> vertices_;
        std::vector<Edge> edges_;
        std::vector<std::pair<int, int>> edge_index_;
        int num_fixed_edges_, num_fixed_vertices_, num_fixed_edge_index_;

    public:
        Graph(const std::vector<Vertex> &vertices = {}, const std::vector<Edge> &edges = {})
            : vertices_(vertices), edges_(edges)
        {
        }

        std::vector<Vertex> vertices() const
        {
            return vertices_;
        }

        std::vector<Edge> edges() const
        {
            return edges_;
        }

        Vertex vertex(int id) const
        {
            if (id >= 0)
                return vertices_[id];
            return vertices_[vertices_.size() + id];
        }

        Edge edge(int start_id, int end_id) const
        {
            if (start_id >= 0)
                return edges_[vertices_[start_id].edge_id(end_id)];
            else
                return edges_[vertices_[vertices_.size() + start_id].edge_id(end_id)];
        }

        std::vector<int> neighbors(int id) const
        {
            if (id >= 0)
                return vertices_[id].neighbors();
            else
                return vertices_[vertices_.size() + id].neighbors();
        }

        void add_vertex(int type, const Point2D &position, int h_direct = UNDIRECTED, int v_direct = UNDIRECTED, int line_type = NONE_LINE)
        {
            if (vertices_.empty())
            {
                vertices_.emplace_back(0, type, position, h_direct, v_direct, line_type);
            }
            else
            {
                vertices_.emplace_back(vertices_.size(), type, position, h_direct, v_direct, line_type);
            }
        }

        void add_edge(int start_id, int end_id, float velocity = 1.0f, float distance = 0.0f, float direction = 0.0f)
        {
            if (edges_.empty())
            {
                edges_.emplace_back(0, vertices_[start_id], vertices_[end_id], velocity, distance, direction);
                edge_index_.push_back(std::make_pair(start_id, end_id));
            }
            else
            {
                edges_.emplace_back(edges_.size(), vertices_[start_id], vertices_[end_id], velocity, distance, direction);
                edge_index_.push_back(std::make_pair(start_id, end_id));
            }
            vertices_[start_id].add_neighbor(end_id, edges_.size() - 1);
        }

        void set_fixed_vertices_and_edges(int num_fixed_vertices, int num_fixed_edges)
        {
            num_fixed_edges_ = num_fixed_edges;
            num_fixed_vertices_ = num_fixed_vertices;
            num_fixed_edge_index_ = edge_index_.size();
        }

        void delete_robot_vertices()
        {
            vertices_.resize(num_fixed_vertices_);
            edges_.resize(num_fixed_edges_);
            edge_index_.resize(num_fixed_edge_index_);
        }

        int num_fixed_edge_index() const { return num_fixed_edge_index_; }
        int num_fixed_vertices() const { return num_fixed_vertices_; }
        int num_fixed_edges() const { return num_fixed_edges_; }

        std::vector<std::pair<int, int>> edge_index() const { return edge_index_; }

        std::vector<std::pair<int, float>> astar_neighbors(int id) const
        {
            std::vector<std::pair<int, float>> neighbors;
            for (int neighbor : vertices_[id].neighbors())
            {
                neighbors.emplace_back(neighbor, euclidean_distance(vertices_[id].position(), vertices_[neighbor].position()));
            }
            return neighbors;
        }

        friend std::ostream &operator<<(std::ostream &os, const Graph &g)
        {
            std::string graph_str;
            for (const Vertex &vertex : g.vertices_)
            {
                std::string type = "unknown";
                switch (vertex.type())
                {
                case WORKING_VERTEX:
                    type = "working";
                    break;
                case STORAGE_VERTEX:
                    type = "storage";
                    break;
                case CHARGING_VERTEX:
                    type = "charging";
                    break;
                case WAITING_VERTEX:
                    type = "waiting";
                    break;
                case LINE_VERTEX:
                    type = "line";
                    break;
                case ROBOT_VERTEX:
                    type = "robot";
                    break;
                }
                graph_str += "ID: " + std::to_string(vertex.id()) + " | Type: " + type + " | Position: x = " + std::to_string(vertex.x()) + ", y = " + std::to_string(vertex.y()) + " | Neighbor: ";
                for (int neighbor : vertex.neighbors())
                {
                    graph_str += std::to_string(neighbor) + " ";
                }
                graph_str += "\n";
            }
            return os << graph_str;
        }
    };

    class GraphZone
    {
    private:
        std::vector<Vertex> vertices_;
        int row_id_, col_id_, zone_id_;
        float length_, width_;
        Point2D center_;

    public:
        GraphZone(int row_id, int col_id, int zone_id, Point2D center, float length, float width)
        {
            row_id_ = row_id;
            col_id_ = col_id;
            zone_id_ = zone_id;
            center_ = center;
            length_ = length;
            width_ = width;
        }

        bool add_vertex(Vertex vertex)
        {
            bool cond1 = vertex.x() >= center_.x() - length_ / 2 && vertex.x() <= center_.x() + length_ / 2;
            bool cond2 = vertex.y() >= center_.y() - width_ / 2 && vertex.y() <= center_.y() + width_ / 2;
            if (cond1 && cond2)
            {
                vertices_.push_back(vertex);
                return true;
            }
            return false;
        }

        Vertex vertex(int id) { return vertices_[id]; }
        std::vector<Vertex> vertices() const { return vertices_; }
        float length() const { return length_; }
        float width() const { return width_; }
        Point2D center() const { return center_; }
        float x() const { return center_.x(); }
        float y() const { return center_.y(); }
        int row_id() const { return row_id_; }
        int col_id() const { return col_id_; }
        int zone_id() const { return zone_id_; }
    };

    class Route
    {
    private:
        int type_;
        std::vector<int> ids_;
        std::vector<Point2D> positions_;
        float astar_cost_;

    public:
        Route(int type = NONE_ROUTE, const std::vector<int> &route_ids = {}, const std::vector<Point2D> &route_positions = {})
            : type_(type), ids_(route_ids), positions_(route_positions)
        {
            if (type == NONE_ROUTE)
                astar_cost_ = 0.0;
            else
                astar_cost_ = calculate_path_cost(route_positions);
        }

        void extend(const std::vector<int> &ids, const std::vector<Point2D> &positions)
        {
            ids_.insert(ids_.end(), ids.begin(), ids.end());
            positions_.insert(positions_.end(), positions.begin(), positions.end());
            astar_cost_ += calculate_path_cost(positions);
        }

        void update(int type, const std::vector<int> &ids, const std::vector<Point2D> &positions)
        {
            type_ = type;
            ids_ = ids;
            positions_ = positions;
            astar_cost_ = calculate_path_cost(positions);
        }

        void clear()
        {
            type_ = NONE_ROUTE;
            positions_.clear();
            astar_cost_ = 0.0;
            ids_.clear();
        }

        bool is_route() const
        {
            return type_ != NONE_ROUTE;
        }

        int type() const
        {
            return type_;
        }

        size_t num_points() const
        {
            return ids_.size();
        }

        float astar_cost() const
        {
            return astar_cost_;
        }

        std::vector<int> ids() const
        {
            return ids_;
        }

        int graph_id(int index) const
        {
            if (index < 0)
                return ids_[ids_.size() + index];
            else
                return ids_[index];
        }

        std::vector<Point2D> positions() const
        {
            return positions_;
        }

        Point2D position(int id) const
        {
            if (id < 0)
                return positions_[positions_.size() + id];
            return positions_[id];
        }

        float x_vertex(int id) const
        {
            if (id < 0)
                return positions_[positions_.size() + id].x();
            return positions_[id].x();
        }

        float y_vertex(int id) const
        {
            if (id < 0)
                return positions_[positions_.size() + id].y();
            return positions_[id].y();
        }

        std::vector<std::vector<float>> to_vector() const
        {
            std::vector<std::vector<float>> positions;
            for (const auto &position : positions_)
            {
                positions.push_back({position.x(), position.y()});
            }
            return positions;
        }

        friend std::ostream &operator<<(std::ostream &os, const Route &route)
        {
            std::string type_str = "NONE_ROUTE";
            if (route.type_ == TO_START)
            {
                type_str = "TO_START";
            }
            else if (route.type_ == TO_TARGET)
            {
                type_str = "TO_TARGET";
            }
            else if (route.type_ == TO_CHARGING)
            {
                type_str = "TO_CHARGING";
            }
            else if (route.type_ == TO_WAITING)
            {
                type_str = "TO_WAITING";
            }

            std::string route_str = "Type: " + type_str + "\n";
            for (size_t i = 0; i < route.ids_.size(); ++i)
            {
                route_str += "Vertex ID: " + std::to_string(route.ids_[i]) +
                             " | Vertex position: x = " + std::to_string(route.positions_[i].x()) + ", y = " + std::to_string(route.positions_[i].y()) + " \n";
            }

            return os << route_str;
        }
    };

    class Task
    {
    private:
        int type_;
        int priority_;
        float mass_;
        Vertex start_;
        Vertex target_;
        Route route_;
        bool is_task_;

    public:
        Task(int type = 0, int priority = 0, float mass = 0.0, const Vertex &start = Vertex(), const Vertex &target = Vertex())
            : type_(type), priority_(priority), mass_(mass), start_(start), target_(target), route_()
        {
            if (start.type() == UNKNOWN_VERTEX || target.type() == UNKNOWN_VERTEX)
                is_task_ = false;
            else
                is_task_ = true;
        }

        void update(int type, int priority, float mass, const Vertex &start, const Vertex &target, const Route &route)
        {
            type_ = type;
            priority_ = priority;
            mass_ = mass;
            start_ = start;
            target_ = target;
            route_ = route;
            is_task_ = true;
        }

        bool is_task() const { return is_task_; }

        void clear_task()
        {
            is_task_ = false;
            priority_ = 0;
            mass_ = 0;
            route_.clear();
        }

        int start_id() const { return start_.id(); }
        int target_id() const { return target_.id(); }
        Vertex start() const { return start_; }
        float start_x() const { return start_.x(); }
        float start_y() const { return start_.y(); }
        Point2D start_position() const { return start_.position(); }
        Vertex target() const { return target_; }
        float target_x() const { return target_.x(); }
        float target_y() const { return target_.y(); }
        Point2D target_position() const { return target_.position(); }
        int type() const { return type_; }
        int priority() const { return priority_; }
        float mass() const { return mass_; }
        Route route() const { return route_; }
        std::vector<Point2D> route_positions() const { return route_.positions(); }
        float route_astar_cost() const { return route_.astar_cost(); }
        void set_route(const Route &route)
        {
            route_ = route;
        }
        friend std::ostream &operator<<(std::ostream &os, const Task &task)
        {
            // std::string type_str = "NONE_ROUTE";
            // if (task.type_ == TO_START)
            //     type_str = "TO_START";
            // else if (task.type_ == TO_TARGET)
            //     type_str = "TO_TARGET";
            // else if (task.type_ == TO_CHARGING)
            //     type_str = "TO_CHARGING";
            // else if (task.type_ == TO_WAITING)
            //     type_str = "TO_WAITING";

            std::string task_str = "Type: " + std::to_string(task.type_) + " | Priority: " + std::to_string(task.priority_) + " | Mass: " + std::to_string(task.mass_);
            task_str += " | Start: x = " + std::to_string(task.start_.x()) + ", y = " + std::to_string(task.start_.y());
            task_str += " | Target: x = " + std::to_string(task.target_.x()) + ", y = " + std::to_string(task.target_.y());
            return os << task_str;
        }
    };

    class AStarItem
    {
    private:
        int current_state_;
        std::vector<int> actions_;
        float cost_;

    public:
        AStarItem(int current_state = 0, std::vector<int> actions = {}, float cost = 0.0) : current_state_(current_state), actions_(actions), cost_(cost) {}
        int current_state() const { return current_state_; }
        std::vector<int> actions() const { return actions_; }
        float cost() const { return cost_; }
        bool operator==(const AStarItem &other) const
        {
            if (current_state_ != other.current_state_)
                return false;
            if (cost_ != other.cost_)
                return false;
            if (actions_.size() != other.actions_.size())
                return false;
            for (int i = 0; i < actions_.size(); i++)
            {
                if (actions_[i] != other.actions_[i])
                    return false;
            }
            return true;
        }
    };

    struct PQEntry
    {
        int priority;
        int count;
        AStarItem item;

        bool operator>(const PQEntry &other) const
        {
            return priority > other.priority;
        }
    };

    class PriorityQueue
    {
    private:
        std::priority_queue<PQEntry, std::vector<PQEntry>, std::greater<PQEntry>> heap;
        int count;

    public:
        PriorityQueue() : count(0)
        {
        }

        void push(AStarItem item, int priority)
        {
            PQEntry entry = {priority, count, item};
            heap.push(entry);
            count++;
        }

        AStarItem pop()
        {
            PQEntry entry = heap.top();
            heap.pop();
            return entry.item;
        }

        bool isEmpty() const
        {
            return heap.empty();
        }

        void update(AStarItem item, int priority)
        {
            std::vector<PQEntry> temp;
            bool found = false;

            while (!heap.empty())
            {
                PQEntry entry = heap.top();
                heap.pop();
                if (entry.item == item)
                {
                    if (entry.priority <= priority)
                    {
                        found = true;
                        break;
                    }
                    entry.priority = priority;
                    entry.count = count++;
                    temp.push_back(entry);
                    found = true;
                    break;
                }
                else
                {
                    temp.push_back(entry);
                }
            }

            if (!found)
            {
                push(item, priority);
            }

            for (const auto &entry : temp)
            {
                heap.push(entry);
            }
        }
    };

    std::pair<std::vector<int>, std::vector<Point2D>> astar_planning(std::shared_ptr<Graph> graph, int start_id, int target_id);
    float astar_planning_cost(std::shared_ptr<Graph> graph, int start_id, int target_id);
    float average_route_velocity(std::shared_ptr<Graph> graph, std::vector<int> route_ids);
    std::vector<std::vector<float>> calculate_average_velocity_and_distance_forward(std::shared_ptr<Graph> graph,
                                                                            std::vector<int> route_ids);
    std::vector<std::vector<float>> calculate_average_velocity_and_distance_backward(std::shared_ptr<Graph> graph,
                                                                            std::vector<int> route_ids);
    bool compare_vertex_by_x(const Vertex &v1, const Vertex &v2);
    bool compare_vertex_by_y(const Vertex &v1, const Vertex &v2);
}
#endif