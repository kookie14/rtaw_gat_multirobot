# pragma once
# ifndef ROBOT_H_
# define ROBOT_H_

# include "utils/type_define.h"

namespace rds
{
    class Goal
    {
    private:
        int type_;               // Goal type (TO TARGET or TO CHARGING or TO_WAITING  or TO_START)
        int graph_id_;           // ID of goal in graph
        Point2DHandle position_; // Position of goal
    public:
        Goal(int type = TO_START, int graph_id = -1, Point2D position = Point2D(0, 0))
            : type_(type), graph_id_(graph_id), position_(std::make_shared<Point2D>(position)) {}
        Goal(int type, int graph_id, Point2DHandle position)
            : type_(type), graph_id_(graph_id), position_(std::make_shared<Point2D>(*position)) {}
        Goal(int type, int graph_id, float x, float y)
            : type_(type), graph_id_(graph_id), position_(std::make_shared<Point2D>(x, y)) {}
        int type() const { return type_; }
        int graph_id() const { return graph_id_; }
        Point2D position() const { return *position_; }
        float x() const { return position_->x(); }
        float y() const { return position_->y(); }
        void set_type(int type) { type_ = type; }
        void set_graph_id(int graph_id) { graph_id_ = graph_id; }
        void set_position(Point2D position) { position_->set(position.x(), position.y()); }
        void set_position(float x, float y) { position_->set(x, y); }
        void set(int type, int graph_id, Point2D position)
        {
            type_ = type;
            graph_id_ = graph_id;
            position_->set(position.x(), position.y());
        }
        void set(int type, int graph_id, Point2DHandle position)
        {
            type_ = type;
            graph_id_ = graph_id;
            position_->set(position->x(), position->y());
        }
        void set(int type, int graph_id, float x, float y)
        {
            type_ = type;
            graph_id_ = graph_id;
            position_->set(x, y);
        }
    };
    typedef std::shared_ptr<Goal> GoalHandle;

    class Robot
    {
    private:
        // Robot parameters
        int id_;
        Pose2DHandle pose_;                  // Current position (x, y) and orientation (theta) of robot
        Point2DHandle velocity_;             // Current velocity (x, y) of robot
        float payload_;                      // Current payload (kg) of robot
        float max_speed_;                    // Maximum speed of robot
        float max_payload_;                  // Maximum payload of robot
        float length_, width_;               // Length and width of robot
        matplot::line_handle shape_visual_;  // Visualize shape and pose of robot
        matplot::labels_handle text_visual_; // Visualize ID of robot
        matplot::line_handle route_visual_; // Visualize route 
        // Allocation and planning parameters
        std::vector<std::array<float, 4>> allocation_state_color_; // Color equivalent with allocation state of robot
        std::vector<std::array<float, 4>> route_color_; // Color equivalent with allocation state of robot
        int graph_id_;                                             // ID of robot in graph
        int allocation_state_;                                     // State of robot for allocation system
        TaskHandle task_;                                          // Task of robot is allocated
        RouteHandle route_;                                        // Route of robot
        int next_point_id_;                                        // ID of next point in route that robot must to go
        GoalHandle global_goal_;                                   // The goal of robot is assigned by task
        GoalHandle local_goal_;                                    // The goal of robot is assigned by planning system
        // Control parameters
        float dt_;                          // Time interval in seconds (default 0.1 seconds - 100 milliseconds)
        int control_state_;                 // State is assigned by local controller
        int num_between_;                   // Number of robots between robot and next point of robot
        int num_same_next_;                 // Number of robots have same next point with robot
        std::vector<int> same_next_ids_;    // Array of robots that have same next point with robot
        Point2DHandle same_point_;                // Same next point with other robots
        int closest_robot_id_;              // ID of closest robot in in a line
        Point2DHandle control_next_point_;  // Next point of robot is assigned by local controller
        float wait_time_for_priority_;      // Wait time to determine the priority of the robot when crossing the next point.
        float wait_time_by_picking_;        // Wait time caused by picking up or dropping off the product.
        std::vector<float> wait_time_list_; // Wait time for priority of robots has same next point
        std::vector<float> dist_to_same_list_; 
        float allocation_reward_;           // Reward for allocation
        bool reward_enabled_;               // Reward enabled for update allocation network
        int allocation_reward_idx_;         // Index of allocation reward
        bool start_reward_count_;           // True if start for calculate reward
        bool done_count_reward_;            // True if done for calculate reward
        float waiting_time_;
        float task_done_;        // Check robot complete task
        bool time_count_enabled_; // Enable for time counter
        int time_count_idx_;
        bool start_count_time_;    // Start count time for complete a task
        bool done_count_time_;     //  Check robot complete task
        float time_complete_task_; // Time complete a task

    public:
        Robot(int id, float dt, Pose2D init_pose, float max_speed, float max_payload, matplot::axes_handle visual,
              float length = 0.98, float width = 0.61);
        void run();
        void run(float max_speed);
        void run(Point2D vel);
        void run(float vx, float vy);

        int id() const { return id_; }

        Pose2D pose() { return *pose_; }
        void set_pose(Pose2D pose) { pose_->set(pose.x(), pose.y(), pose.theta()); }
        void set_pose(Point2D position, float theta) { pose_->set(position.x(), position.y(), theta); }
        void set_pose(float x, float y, float theta) { pose_->set(x, y, theta); }
        Point2D position() { return pose_->position(); }
        void set_position(Point2D position)
        {
            pose_->set_x(position.x());
            pose_->set_y(position.y());
        }
        void set_position(float x, float y)
        {
            pose_->set_x(x);
            pose_->set_y(y);
        }
        float x() { return pose_->x(); }
        float y() { return pose_->y(); }
        float theta() { return pose_->theta(); }
        void set_theta(float theta) { pose_->set_theta(theta); }

        Point2D velocity() { return *velocity_; }
        void set_velocity(Point2D velocity);
        void set_velocity(float vx, float vy);
        void set_velocity(Point2D velocity, float max_speed);
        void set_velocity(float vx, float vy, float max_speed);
        Point2D limit_velocity(Point2D velocity);
        Point2D limit_velocity(Point2D velocity, float max_speed);
        std::pair<float, float> limit_velocity(float vx, float vy);
        std::pair<float, float> limit_velocity(float vx, float vy, float max_speed);
        float vx() { return velocity_->x(); }
        void set_vx(float vx) { velocity_->set_x(vx); }
        float vy() { return velocity_->y(); }
        void set_vy(float vy) { velocity_->set_y(vy); }

        float payload() { return payload_; }
        void set_payload(float payload) { payload_ = payload; }
        void plus_payload(float payload) { payload_ += payload; }
        float rest_payload() { return max_payload_ - payload_; }

        float length() { return length_; }
        void set_length(float length) { length_ = length; }
        float width() { return width_; }
        void set_width(float width) { width_ = width; }

        int graph_id() { return graph_id_; }
        void set_graph_id(int graph_id) { graph_id_ = graph_id; }

        int allocation_state() { return allocation_state_; }
        void set_allocation_state(int allocation_state) { allocation_state_ = allocation_state; }

        bool has_task() { return task_->is_task(); }
        void clear_task() { task_->clear_task(); }
        int start_id() { return task_->start_id(); }
        int target_id() { return task_->target_id(); }
        Point2D start_position() { return task_->start_position(); }
        float start_x() { return task_->start_x(); }
        float start_y() { return task_->start_y(); }
        Point2D target_position() { return task_->target_position(); }
        float target_x() { return task_->target_x(); }
        float target_y() { return task_->target_y(); }
        void set_task(Task task);
        int task_done() { return task_done_; }
        void set_task_done(bool done) { task_done_ = done; }

        bool has_route() { return route_->is_route(); }
        int route_type() { return route_->type(); }
        int next_point_id() { return next_point_id_; }
        void plus_next_point_id() { next_point_id_ += 1; }
        std::vector<int> route_ids() { return route_->ids(); }
        int route_graph_id(int index) { return route_->graph_id(index); }
        std::vector<Point2D> route_positions() { return route_->positions(); }
        Point2D route_position(int index) { return route_->position(index); }
        Point2D next_point_in_route() { return route_->position(next_point_id_); }
        bool route_done()
        {
            if (route_->is_route())
            {
                if (is_same_point(pose_->position(), route_->position(-1)))
                {
                    return true;
                }
            }
            return false;
        }
        void clear_route() { route_->clear(); }
        void set_route(Route route);
        void set_route(int type, std::vector<int> ids, std::vector<Point2D> positions);
        void extend_route(Route route);
        void extend_route(std::vector<int> ids, std::vector<Point2D> positions);

        Point2D global_goal_position() { return global_goal_->position(); }
        int global_goal_id() { return global_goal_->graph_id(); }
        int global_goal_type() { return global_goal_->type(); }
        float global_goal_x() { return global_goal_->x(); }
        float global_goal_y() { return global_goal_->y(); }
        void set_global_goal(int type, int graph_id, Point2D position)
        {
            global_goal_->set(type, graph_id, position);
        }
        void set_global_goal(int type, int graph_id, Point2DHandle position)
        {
            global_goal_->set(type, graph_id, position);
        }
        void set_global_goal(int type, int graph_id, float x, float y)
        {
            global_goal_->set(type, graph_id, x, y);
        }

        Point2D local_goal_position() { return local_goal_->position(); }
        int local_goal_id() { return local_goal_->graph_id(); }
        float local_goal_x() { return local_goal_->x(); }
        float local_goal_y() { return local_goal_->y(); }
        void set_local_goal(int type, int graph_id, Point2D position)
        {
            local_goal_->set(type, graph_id, position);
        }
        void set_local_goal(int type, int graph_id, Point2DHandle position)
        {
            local_goal_->set(type, graph_id, position);
        }
        void set_local_goal(int type, int graph_id, float x, float y)
        {
            local_goal_->set(type, graph_id, x, y);
        }

        int control_state() { return control_state_; }
        void set_control_state(int control_state) { control_state_ = control_state; }

        int num_between() { return num_between_; }
        void set_num_between(int num_between) { num_between_ = num_between; }

        int num_same_next() { return num_same_next_; }
        void set_num_same_next(int num_same_next) { num_same_next_ = num_same_next; }

        Point2D same_point() { return *same_point_; }
        void set_same_point(Point2D point) { same_point_->set(point.x(), point.y()); }
        void set_same_point(float x, float y) { same_point_->set(x, y); }

        std::vector<int> same_next_ids() { return same_next_ids_; }
        void set_same_next_ids(std::vector<int> same_next_ids) { same_next_ids_ = same_next_ids; }

        int closest_robot_id() { return closest_robot_id_; }
        void set_closest_robot_id(int closest_robot_id) { closest_robot_id_ = closest_robot_id; }

        Point2D control_next_point() { return *control_next_point_; }
        void set_control_next_point(Point2D p) { control_next_point_->set(p.x(), p.y()); }
        void set_control_next_point(float x, float y) { control_next_point_->set(x, y); }

        float waiting_time() const { return waiting_time_; }
        void set_waiting_time(float time) { waiting_time_ = time; }

        float get_target_time() const { return task_->route().astar_cost() * dt_; }

        float wait_time_for_priority() { return wait_time_for_priority_; }
        void set_wait_time_for_priority(float wait_time_for_priority) { wait_time_for_priority_ = wait_time_for_priority; }
        void plus_wait_time_for_priority() { wait_time_for_priority_ += dt_; }
        void reset_wait_time_for_priority() { wait_time_for_priority_ = 0.0; }

        std::vector<float> wait_time_list() {return wait_time_list_; }
        std::vector<float> dist_to_same_list() {return dist_to_same_list_; }
        void set_dist_to_same_list(std::vector<float> dist_to_same_list) {dist_to_same_list_ = dist_to_same_list; }
        void set_wait_time_list(std::vector<float> wait_time_list) {wait_time_list_ = wait_time_list;}

        float wait_time_by_picking() { return wait_time_by_picking_; }
        void plus_wait_time_by_picking() { wait_time_by_picking_ += dt_; }
        void set_wait_time_by_picking(int wait_time_by_picking) { wait_time_by_picking_ = wait_time_by_picking; }
        void reset_wait_time_by_picking() { wait_time_by_picking_ = 0.0; }

        float allocation_reward() { return allocation_reward_; }
        void set_allocation_reward(float allocation_reward) { allocation_reward_ = allocation_reward; }
        void reset_allocation_reward() { allocation_reward_ = 0.0f; }

        int allocation_reward_idx() { return allocation_reward_idx_; }
        void plus_allocation_reward() { allocation_reward_ += dt_; }
        void set_allocation_reward_idx(int allocation_reward_idx)
        {
            allocation_reward_idx_ = allocation_reward_idx;
            start_reward_count_ = true;
            allocation_reward_ = 0.0;
            done_count_reward_ = false;
        }

        bool reward_enabled() { return reward_enabled_; }
        void set_reward_enabled(bool enabled) { reward_enabled_ = enabled; }

        bool start_reward_count() { return start_reward_count_; }
        void set_start_reward_count(bool start_reward_count) { start_reward_count_ = start_reward_count; }

        bool done_count_reward() { return done_count_reward_; }
        void set_done_count_reward(bool done_count_reward) { done_count_reward_ = done_count_reward; }

        int time_count_idx() { return time_count_idx_; }
        void plus_time_completed_task() {time_complete_task_ += dt_;}
        void set_time_count_idx(int time_count_idx)
        {
            time_count_idx_ = time_count_idx;
            start_count_time_ = true;
            done_count_time_ = false;
            time_complete_task_ = 0.0f;
        }

        bool time_count_enabled() {return time_count_enabled_;}
        void set_time_count_enabled(bool time_count_enabled) {time_count_enabled_ = time_count_enabled;}

        bool start_count_time() {return start_count_time_;}
        void set_start_count_time(bool start_count_time) {start_count_time_ = start_count_time;}

        bool done_count_time() {return done_count_time_;}
        void set_done_count_time(bool done_count_time) {done_count_time_ = done_count_time;}

        float time_complete_task() {return time_complete_task_;}
        void plus_time_complete_task() {time_complete_task_+= 1;}
        void reset_time_complete_task() {time_complete_task_ = 0.0f;}

        bool local_goal_reached()
        {
            return is_same_point(local_goal_->position(), pose_->position());
        }

        bool global_goal_reached()
        {
            return is_same_point(global_goal_->position(), pose_->position());
        }

        bool task_is_done();
        void start_is_done();
        bool wait_for_picking_or_dropping(float waiting_time);

        void visualize(GraphHandle graph);
    };
    typedef std::shared_ptr<Robot> RobotHandle;
    typedef std::vector<RobotHandle> SwarmHandle;
}
# endif