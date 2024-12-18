#include "robot/robot.h"

namespace rds
{
    Robot::Robot(int id, float dt, Pose2D init_pose, float max_speed, float max_payload, matplot::axes_handle visual,
                 float length, float width)
    {
        // FREE, ON_WAY_TO_START, ON_WAY_TO_TARGET, ON_WAY_TO_WAITING, ON_WAY_TO_CHARGING, PICKING_UP, BUSY, LOW_BATTERY
        // CHARGING, AVOIDANCE, FOLLOWING, DIE
        allocation_state_color_ = {GREEN, GOLD, GOLD, CHOCOLATE, NAVY, CYAN, ORANGE, RED, BLUE, PURPLE, BROWN, BLACK};
        route_color_ = {RED, GREEN, BLUE, CYAN, ORANGE, GOLD, YELLOW, ORANGE, PURPLE, BROWN};
        id_ = id;
        dt_ = dt;
        pose_ = std::make_shared<Pose2D>(init_pose.x(), init_pose.y(), init_pose.theta());
        velocity_ = std::make_shared<Point2D>(0.0f, 0.0f);
        payload_ = 0.0;
        max_speed_ = max_speed;
        max_payload_ = max_payload;
        length_ = length;
        width_ = width;
        allocation_state_ = FREE;
        V2d robot_shape = vector_to_plot_2d(calculate_rectangle_coordinate(pose_->x(), pose_->y(), pose_->theta(), length_, width_));
        // V1d x_route, y_route;
        // x_route.push_back(init_pose.x());
        // x_route.push_back(init_pose.x());
        // y_route.push_back(init_pose.y());
        // y_route.push_back(init_pose.y());
        // route_visual_ = visual->plot(x_route, y_route);
        // route_visual_->line_width(3);
        // route_visual_->color(allocation_state_color_[id%10]);

        shape_visual_ = visual->fill(robot_shape[0], robot_shape[1]);
        shape_visual_->color(allocation_state_color_[allocation_state_]);
        text_visual_ = visual->text(pose_->x(), pose_->y(), std::to_string(id_));
        text_visual_->color(BLACK);
        text_visual_->font_size(10);
        
        graph_id_ = 0;
        task_ = std::make_shared<Task>();
        route_ = std::make_shared<Route>();
        next_point_id_ = 1;
        global_goal_ = std::make_shared<Goal>();
        local_goal_ = std::make_shared<Goal>();
        // Control parameters
        control_state_ = WAIT;
        num_between_ = 0;
        num_same_next_ = 0;
        closest_robot_id_ = 0;
        same_point_ = std::make_shared<Point2D>();
        control_next_point_ = std::make_shared<Point2D>(init_pose.position());
        wait_time_for_priority_ = 0.0f;
        wait_time_by_picking_ = 0.0f;
        allocation_reward_ = 0.0f;
        allocation_reward_idx_ = 0;
        start_reward_count_ = false;
        done_count_reward_ = false;
        task_done_ = false;
        time_count_idx_ = 0;
        time_count_enabled_ = false;
        start_count_time_ = true;
        done_count_time_ = false;
        time_complete_task_ = 0.0f;
    }

    void Robot::run()
    {
        if (control_state_ == TO_NEXT_POINT || control_state_ == TO_WAIT_POINT)
            reset_wait_time_for_priority();

        if (control_state_ == WAIT && allocation_state_ != PICKING_UP)
            plus_wait_time_for_priority();

        if (control_state_ == WAIT)
        {
            velocity_->set(0.0f, 0.0f);
        }
        else
        {
            set_velocity((*control_next_point_ - pose_->position()) / dt_);
            pose_->set_x(pose_->x() + velocity_->x() * dt_);
            pose_->set_y(pose_->y() + velocity_->y() * dt_);
        }
        if (next_point_id_ < route_->ids().size() - 1)
            pose_->set_theta(angle_by_two_point(route_->position(next_point_id_ - 1), control_next_point()));
        else
            pose_->set_theta(angle_by_two_point(control_next_point(), route_->position(next_point_id_ - 1)));

        if (is_same_point(pose_->position(), next_point_in_route()) && next_point_id_ < route_->ids().size() - 1)
        {
            plus_next_point_id();
        }
    }

    void Robot::run(float max_speed)
    {
        if (control_state_ == TO_NEXT_POINT || control_state_ == TO_WAIT_POINT)
            reset_wait_time_for_priority();

        if (control_state_ == WAIT && allocation_state_ != PICKING_UP)
            plus_wait_time_for_priority();

        if (control_state_ == WAIT)
        {
            velocity_->set(0.0f, 0.0f);
        }
        else
        {
            set_velocity((*control_next_point_ - pose_->position()) / dt_, max_speed);
            pose_->set_x(pose_->x() + velocity_->x() * dt_);
            pose_->set_y(pose_->y() + velocity_->y() * dt_);
        }
        if (next_point_id_ < route_->ids().size() - 1)
            pose_->set_theta(angle_by_two_point(route_->position(next_point_id_ - 1), control_next_point()));
        else
            pose_->set_theta(angle_by_two_point(control_next_point(), route_->position(next_point_id_ - 1)));

        if (is_same_point(pose_->position(), next_point_in_route()) && next_point_id_ < route_->ids().size() - 1)
        {
            plus_next_point_id();
        }
    }

    void Robot::run(Point2D vel)
    {
        set_velocity(limit_velocity(vel));
        pose_->set_x(pose_->x() + velocity_->x() * dt_);
        pose_->set_y(pose_->y() + velocity_->y() * dt_);
        pose_->set_theta(atan2(velocity_->y(), velocity_->x()));
    }

    void Robot::run(float vx, float vy)
    {
        std::pair<float, float> vel = limit_velocity(vx, vy);
        set_velocity(vel.first, vel.second);
        pose_->set_x(pose_->x() + velocity_->x() * dt_);
        pose_->set_y(pose_->y() + velocity_->y() * dt_);
        pose_->set_theta(atan2(velocity_->y(), velocity_->x()));
    }

    void Robot::set_velocity(Point2D velocity)
    {
        Point2D limited_vel = limit_velocity(velocity);
        velocity_->set(limited_vel.x(), limited_vel.y());
    }

    void Robot::set_velocity(Point2D velocity, float max_speed)
    {
        Point2D limited_vel = limit_velocity(velocity, max_speed);
        velocity_->set(limited_vel.x(), limited_vel.y());
    }

    void Robot::set_velocity(float vx, float vy, float max_speed)
    {
        std::pair<float, float> limited_vel = limit_velocity(vx, vy, max_speed);
        velocity_->set(limited_vel.first, limited_vel.second);
    }

    void Robot::set_velocity(float vx, float vy)
    {
        std::pair<float, float> limited_vel = limit_velocity(vx, vy);
        velocity_->set(limited_vel.first, limited_vel.second);
    }

    Point2D Robot::limit_velocity(Point2D velocity)
    {
        float speed = hypot(velocity.x(), velocity.y());
        if (speed > max_speed_)
        {
            float vx, vy;
            vx = (velocity.x() / speed) * max_speed_;
            vy = (velocity.y() / speed) * max_speed_;
            return Point2D(vx, vy);
        }
        return velocity;
    }

    Point2D Robot::limit_velocity(Point2D velocity, float max_speed)
    {
        float speed = hypot(velocity.x(), velocity.y());
        if (speed > max_speed)
        {
            float vx, vy;
            vx = (velocity.x() / speed) * max_speed;
            vy = (velocity.y() / speed) * max_speed;
            return Point2D(vx, vy);
        }
        return velocity;
    }

    std::pair<float, float> Robot::limit_velocity(float vx, float vy)
    {
        float speed = hypot(vx, vy);
        if (speed > max_speed_)
        {
            float limited_vx, limited_vy;
            limited_vx = (vx / speed) * max_speed_;
            limited_vy = (vy / speed) * max_speed_;
            return std::pair<float, float>(limited_vx, limited_vy);
        }
        return std::pair<float, float>(vx, vy);
    }
    std::pair<float, float> Robot::limit_velocity(float vx, float vy, float max_speed)
    {
        float speed = hypot(vx, vy);
        if (speed > max_speed_)
        {
            float limited_vx, limited_vy;
            limited_vx = (vx / speed) * max_speed;
            limited_vy = (vy / speed) * max_speed;
            return std::pair<float, float>(limited_vx, limited_vy);
        }
        return std::pair<float, float>(vx, vy);
    }

    void Robot::set_task(Task task)
    {
        task_->update(task.type(), task.priority(), task.mass(), task.start(), task.target(), task.route());
        payload_ += task.mass();
        set_global_goal(TO_START, task.start_id(), task.start_position());
        reset_wait_time_by_picking();
        task_done_ = false;
    }

    void Robot::set_route(Route route)
    {
        route_->update(route.type(), route.ids(), route.positions());
        next_point_id_ = 1;
        reset_wait_time_by_picking();
    }

    void Robot::set_route(int type, std::vector<int> ids, std::vector<Point2D> positions)
    {
        route_->update(type, ids, positions);
        next_point_id_ = 1;
        reset_wait_time_by_picking();
    }

    void Robot::extend_route(Route route)
    {
        std::vector<int> ids = route.ids();
        std::vector<Point2D> positions = route.positions();
        if (is_same_point(position(), positions[0]))
        {
            positions.erase(positions.begin());
            positions.erase(positions.begin());
            ids.erase(ids.begin());
            ids.erase(ids.begin());
        }
        route_->extend(ids, positions);
    }

    void Robot::extend_route(std::vector<int> ids, std::vector<Point2D> positions)
    {
        if (is_same_point(position(), positions[0]))
        {
            positions.erase(positions.begin());
            positions.erase(positions.begin());
            ids.erase(ids.begin());
            ids.erase(ids.begin());
        }
        route_->extend(ids, positions);
    }

    bool Robot::task_is_done()
    {
        if (has_task())
        {
            if (is_same_point(task_->target_position(), pose_->position()))
            {
                set_allocation_state(FREE);
                return true;
            }
        }
        return false;
    }

    void Robot::start_is_done()
    {
        if (has_task())
        {
            if (is_same_point(task_->start_position(), pose_->position()))
            {
                set_global_goal(TO_TARGET, task_->target_id(), task_->target_position());
            }
        }
    }

    bool Robot::wait_for_picking_or_dropping(float waiting_time)
    {
        if (wait_time_by_picking_ < waiting_time)
        {
            allocation_state_ = PICKING_UP;
            wait_time_by_picking_ += dt_;
            return false;
        }
        else
        {
            allocation_state_ = BUSY;
            return true;
        }
    }

    void Robot::visualize(GraphHandle graph)
    {
        V2d robot_shape = vector_to_plot_2d(calculate_rectangle_coordinate(pose_->x(), pose_->y(), pose_->theta(), length_, width_));
        shape_visual_->x_data(robot_shape[0]);
        shape_visual_->y_data(robot_shape[1]);
        shape_visual_->color(allocation_state_color_[allocation_state_]);
        text_visual_->x({pose_->x()});
        text_visual_->y({pose_->y()});
        // std::pair<std::vector<int>, std::vector<Point2D>> route = astar_planning(graph, graph_id_, global_goal_id());
        // if (route.second.size() > 0)
        // {
        //     V1d x_route, y_route;
        //     for (int i = 0; i < route.second.size(); i++)
        //     {
        //         x_route.push_back(route.second[i].x());
        //         y_route.push_back(route.second[i].y());
        //     }
        //     route_visual_->x_data(x_route);
        //     route_visual_->y_data(y_route);
        // }
        // else
        // {
        //     V1d x_route, y_route;
        //     x_route.push_back(pose_->x());
        //     x_route.push_back(pose_->x());
        //     y_route.push_back(pose_->x());
        //     y_route.push_back(pose_->x());
        //     route_visual_->x_data(x_route);
        //     route_visual_->y_data(y_route);
        // }
    }
}