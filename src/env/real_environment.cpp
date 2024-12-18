#include "env/real_environment.h"

namespace rds
{
    RealEnvironment::RealEnvironment(std::string data_path, int num_swarm_, float robot_max_speed, float robot_max_payload,
                                    matplot::axes_handle visual, bool show_edge)
    {
        dt_ = 0.1;
        factory_ = std::make_shared<RealReconstruction>(data_path);
        factory_->visualize(visual, show_edge);
        graph_ = std::make_shared<Graph>(factory_->graph());
        Vertices homes = factory_->waiting_vertices();
        num_swarm_ = 6;
        swarm_.push_back(std::make_shared<Robot>(0, dt_, Pose2D(147.75, 6.50, 0.0), robot_max_speed, robot_max_payload, visual, 0.98, 0.61));
        swarm_.push_back(std::make_shared<Robot>(1, dt_, Pose2D(147.75, 7.60, 0.0), robot_max_speed, robot_max_payload, visual, 0.98, 0.61));
        swarm_.push_back(std::make_shared<Robot>(2, dt_, Pose2D(207.25, 23.50, M_PI), robot_max_speed, robot_max_payload, visual, 0.98, 0.61));
        swarm_.push_back(std::make_shared<Robot>(3, dt_, Pose2D(165.75, 26.50, 0), robot_max_speed, robot_max_payload, visual, 0.98, 0.61));
        swarm_.push_back(std::make_shared<Robot>(4, dt_, Pose2D(140.00, 33.75, M_PI_2), robot_max_speed, robot_max_payload, visual, 0.98, 0.61));
        swarm_.push_back(std::make_shared<Robot>(5, dt_, Pose2D(147.00, 26.25, -M_PI_2), robot_max_speed, robot_max_payload, visual, 0.98, 0.61));
        d_safe_junction_ = swarm_[0]->length() + 0.01;
        d_safe_closest_ = swarm_[0]->length() + 0.2;
        // std::cout << *graph_ << std::endl;
    }

    void RealEnvironment::controller(float wait_time)
    {
        std::vector<bool> traffic_control;
        for (int i = 0; i < swarm_.size(); i++)
        {
            if (task_assign_control_state(swarm_[i], wait_time) == true)
            {
                swarm_[i]->set_control_state(WAIT);
                traffic_control.push_back(false);
            }
            else
            {
                calculate_same_and_between(swarm_[i]);
                traffic_control.push_back(true);
            }
        }
        std::vector<int> id_list;
        for (int i = 0; i < swarm_.size(); i++)
            id_list.push_back(i);
        std::sort(id_list.begin(), id_list.end(), [this](int a, int b){ return swarm_[a]->num_between() < swarm_[b]->num_between();});
        for (int id : id_list)
        {
            if (traffic_control[id] == true)
            {
                traffic_assign_control_state(swarm_[id]);
            }
        }
    }

    bool RealEnvironment::task_assign_control_state(RobotHandle robot, float wait_time)
    {
        if (robot->has_task() == false)
        {
            return true;
        }
        else
        {
            if (robot->has_route() == false)
                return true;
            else
            {
                if (robot->route_done() == false)
                {
                    if (robot->start_reward_count())
                        robot->plus_allocation_reward();
                    if (robot->route_type() == TO_START)
                        robot->set_allocation_state(ON_WAY_TO_START);
                    if (robot->route_type() == TO_TARGET)
                        robot->set_allocation_state(ON_WAY_TO_TARGET);
                    if (robot->route_type() == TO_WAITING)
                        robot->set_allocation_state(ON_WAY_TO_WAITING);
                    if (robot->route_type() == TO_CHARGING)
                        robot->set_allocation_state(ON_WAY_TO_CHARGING);
                    return false;
                }
                else
                {
                    if (robot->wait_for_picking_or_dropping(wait_time) == false)
                        return true;
                    else
                    {
                        if (robot->global_goal_type() == TO_START)
                        {
                            robot->set_start_reward_count(false);
                            robot->set_done_count_reward(true);
                            robot->start_is_done();
                            robot->clear_route();
                            return true;
                        }
                        else if (robot->global_goal_type() == TO_TARGET)
                        {
                            robot->set_allocation_state(FREE);
                            robot->clear_route();
                            robot->clear_task();
                            return true;
                        }
                        else return true;
                    }
                }
            }
        }
    }

    void RealEnvironment::traffic_assign_control_state(RobotHandle robot)
    {
        if (robot->num_between() == 0)
        {
            if (robot->num_same_next() == 0)
            {
                if (vertex_is_safe(robot, robot->next_point_in_route()))
                {
                    robot->set_control_next_point(robot->next_point_in_route());
                    robot->set_control_state(TO_NEXT_POINT);
                }
                else
                {
                    Point2D wp = calculate_wait_point(robot, robot->next_point_in_route(), d_safe_junction_);
                    if (is_same_point(robot->position(), wp))
                        robot->set_control_state(WAIT);
                    else
                    {
                        robot->set_control_state(TO_WAIT_POINT);
                        robot->set_control_next_point(wp);
                    }
                }
            }
            else
                set_control_state_to_pass_same_point(robot);
        }
        else
        {
            RobotHandle closest_robot = swarm_[robot->closest_robot_id()];
            if (robot->num_same_next() == 0)
            {
                if (point_is_in_segment(robot->position(), closest_robot->position(), robot->next_point_in_route()))
                {
                    robot->set_control_next_point(robot->next_point_in_route());
                    robot->set_control_state(TO_NEXT_POINT);
                }
                else
                {
                    Point2D wp = calculate_wait_point(robot, swarm_[robot->closest_robot_id()]->position(), d_safe_closest_);
                    if (is_same_point(robot->position(), wp))
                    {
                        robot->set_control_state(WAIT);
                    }
                    else
                    {
                        robot->set_control_state(TO_NEXT_POINT);
                        robot->set_control_next_point(wp);
                    }
                }
            }
            else
            {
                if (point_is_in_segment(robot->position(), closest_robot->position(), robot->same_point()))
                {
                    set_control_state_to_pass_same_point(robot);
                }
                else
                {
                    Point2D wp = calculate_wait_point(robot, swarm_[robot->closest_robot_id()]->position(), d_safe_closest_);
                    if (is_same_point(robot->position(), wp))
                    {
                        robot->set_control_state(WAIT);
                    }
                    else
                    {
                        robot->set_control_state(TO_WAIT_POINT);
                        robot->set_control_next_point(wp);
                    }
                }
            }
        }
        // std::cout << "Robot " << robot->id() << ": next point id: " << robot->next_point_id() << ", next point position: " << robot->next_point_in_route() << ", control state: " << robot->control_state() << ", control next point: " << robot->control_next_point() << std::endl;
        // std::cout << "Robot position: " << robot->position() << ", Num same: " << robot->num_same_next() << ", Num between: " << robot->num_between() << ", Closest id: " << robot->closest_robot_id() << std::endl;
        if (robot->next_point_id() == 1)
            robot->run();
        else
        {
            robot->run(graph_->edge(robot->route_graph_id(robot->next_point_id() - 1), robot->route_graph_id(robot->next_point_id())).velocity());
        }

    }

    void RealEnvironment::set_control_state_to_pass_same_point(RobotHandle robot)
    {
        if (euclidean_distance(robot->position(), robot->same_point()) < d_safe_junction_)
        {
            robot->set_control_next_point(robot->same_point());
            robot->set_control_state(TO_WAIT_POINT);
        }
        else
        {
            std::vector<float> wait_time_list = {robot->wait_time_for_priority()}, dist_to_same_list = {euclidean_distance(robot->position(), robot->same_point())};
            for (int id : robot->same_next_ids())
            {
                wait_time_list.push_back(swarm_[id]->wait_time_for_priority());
                dist_to_same_list.push_back(euclidean_distance(swarm_[id]->position(), robot->same_point()));
            }
            float max_wait_time = *std::max_element(wait_time_list.begin(), wait_time_list.end());
            float min_dist = *std::min_element(dist_to_same_list.begin(), dist_to_same_list.end());

            if (std::find(wait_time_list.begin(), wait_time_list.end(), max_wait_time) == wait_time_list.begin())
            {
                if (std::count(wait_time_list.begin(), wait_time_list.end(), max_wait_time) == 1)
                {
                    if (vertex_is_safe(robot, robot->same_point()))
                    {
                        robot->set_control_state(TO_NEXT_POINT);
                        robot->set_control_next_point(robot->same_point());
                    }
                    else
                    {
                        Point2D wp = calculate_wait_point(robot, robot->same_point(), d_safe_junction_);
                        if (is_same_point(robot->position(), wp))
                        {
                            robot->set_control_state(WAIT);
                        }
                        else
                        {
                            robot->set_control_state(TO_WAIT_POINT);
                            robot->set_control_next_point(wp);
                        }
                    }
                }
                else
                {
                    if (std::find(dist_to_same_list.begin(), dist_to_same_list.end(), min_dist) == dist_to_same_list.begin())
                    {
                        if (vertex_is_safe(robot, robot->same_point()))
                        {
                            robot->set_control_state(TO_NEXT_POINT);
                            robot->set_control_next_point(robot->same_point());
                        }
                        else
                        {
                            Point2D wp = calculate_wait_point(robot, robot->same_point(), d_safe_junction_);
                            if (is_same_point(robot->position(), wp))
                            {
                                robot->set_control_state(WAIT);
                            }
                            else
                            {
                                robot->set_control_state(TO_WAIT_POINT);
                                robot->set_control_next_point(wp);
                            }
                        }
                    }
                    else
                    {
                        auto wp = calculate_wait_point(robot, robot->same_point(), d_safe_junction_);
                        if (is_same_point(robot->position(), wp))
                        {
                            robot->set_control_state(WAIT);
                        }
                        else
                        {
                            robot->set_control_state(TO_WAIT_POINT);
                            robot->set_control_next_point(wp);
                        }
                    }
                }
            }
            else
            {
                auto wp = calculate_wait_point(robot, robot->same_point(), d_safe_junction_);
                if (is_same_point(robot->position(), wp))
                {
                    robot->set_control_state(WAIT);
                }
                else
                {
                    robot->set_control_state(TO_WAIT_POINT);
                    robot->set_control_next_point(wp);
                }
            }
        }
    }

    void RealEnvironment::calculate_same_and_between(RobotHandle robot)
    {
        std::vector<int> same_line_list = {robot->id()};
        std::vector<int> same_vertex_list;
        float min_same_dist = std::numeric_limits<float>::infinity();
        bool angle_cond, cond1, cond2, cond3, cond4;
        for (const auto &other : swarm_)
        {
            if (other->id() == robot->id())
                continue;
            std::pair<bool, Point2D> intersect = check_segment_intersection(robot->position(), robot->next_point_in_route(),
                                                                            other->position(), other->next_point_in_route());
            angle_cond = angle_by_two_point(other->position(), other->next_point_in_route()) == angle_by_two_point(robot->position(), robot->next_point_in_route());
            // std::cout << angle_by_two_point(other->position(), other->next_point_in_route()) << " " << angle_by_two_point(robot->position(), robot->next_point_in_route()) << std::endl;
            if (intersect.first)
            {
                cond1 = min_same_dist >= euclidean_distance(robot->position(), intersect.second);
                cond2 = point_is_in_segment(robot->position(), robot->next_point_in_route(), intersect.second);
                cond3 = point_is_in_segment(other->position(), other->next_point_in_route(), intersect.second);
                cond4 = !is_same_point(robot->position(), intersect.second);
                if (cond1 && cond2 && cond3 && cond4 && !angle_cond)
                {
                    min_same_dist = euclidean_distance(robot->position(), intersect.second);
                    robot->set_same_point(intersect.second);
                    same_vertex_list.push_back(other->id());
                }
            }
            if (angle_cond && is_collinear(robot->position(), robot->next_point_in_route(), other->next_point_in_route()))
            {
                same_line_list.push_back(other->id());
            }
        }
        float angle = angle_by_two_point(robot->position(), robot->next_point_in_route());
        std::sort(same_line_list.begin(), same_line_list.end(), [&](int id1, int id2)
                  { return swarm_[id1]->x() < swarm_[id2]->x(); });

        if (-M_PI / 2 - 0.01 <= angle && angle <= -M_PI / 2 + 0.01)
        {
            std::sort(same_line_list.begin(), same_line_list.end(), [&](int id1, int id2)
                      { return swarm_[id1]->y() > swarm_[id2]->y(); });
        }
        if (M_PI / 2 - 0.01 <= angle && angle <= M_PI / 2 + 0.01)
        {
            std::sort(same_line_list.begin(), same_line_list.end(), [&](int id1, int id2)
                      { return swarm_[id1]->y() < swarm_[id2]->y(); });
        }
        if (M_PI - 0.01 <= std::fabs(normalize_angle(angle)) && std::fabs(normalize_angle(angle)) <= M_PI + 0.01)
        {
            std::sort(same_line_list.begin(), same_line_list.end(), [&](int id1, int id2)
                      { return swarm_[id1]->x() > swarm_[id2]->x(); });
        }
        if (-0.01 <= angle && angle <= 0.01)
        {
            std::sort(same_line_list.begin(), same_line_list.end(), [&](int id1, int id2)
                      { return swarm_[id1]->x() < swarm_[id2]->x(); });
        }

        auto it = std::find(same_line_list.begin(), same_line_list.end(), robot->id());
        if (it != same_line_list.end())
        {
            std::vector<int> between_id_filtered(it + 1, same_line_list.end());
            robot->set_num_between(between_id_filtered.size());
            if (!between_id_filtered.empty())
            {
                robot->set_closest_robot_id(between_id_filtered[0]);
            }
        }

        if (!same_vertex_list.empty())
        {
            std::vector<int> same_id_filtered = filter_same_vertex_list(robot, same_vertex_list);
            robot->set_same_next_ids(same_id_filtered);
            robot->set_num_same_next(same_id_filtered.size());
        }
        else
        {
            robot->set_num_same_next(same_vertex_list.size());
            robot->set_same_next_ids(same_vertex_list);
        }
        // std::cout << "Num between: " << robot->num_between() << " Num same : " << robot->num_same_next() << std::endl;
    }

    std::vector<int> RealEnvironment::filter_same_vertex_list(RobotHandle robot, std::vector<int> same_vertex_list)
    {
        if (same_vertex_list.size() == 1)
            return same_vertex_list;

        std::vector<int> same_id_filtered;
        std::vector<std::vector<float>> list_1, list_2, list_3;
        Point2D robot_same_point = robot->same_point();

        for (int id : same_vertex_list)
        {
            std::pair<int, Point2D> intersect = check_segment_intersection(robot->position(), robot->next_point_in_route(),
                                                                           swarm_[id]->position(), swarm_[id]->next_point_in_route());

            if (is_same_point(robot_same_point, intersect.second))
            {
                std::vector<float> data = {static_cast<float>(id), euclidean_distance(swarm_[id]->position(), robot_same_point),
                                           angle_by_two_point(swarm_[id]->position(), robot_same_point)};
                if (list_1.empty() || std::fabs(calculate_difference_angle(data[2], list_1[0][2])) < M_PI / 12)
                {
                    list_1.push_back(data);
                }
                else if (list_2.empty() || std::fabs(calculate_difference_angle(data[2], list_2[0][2])) < M_PI / 12)
                {
                    list_2.push_back(data);
                }
                else
                {
                    list_3.push_back(data);
                }
            }
        }

        auto sort_and_append = [&same_id_filtered](std::vector<std::vector<float>> &list)
        {
            if (!list.empty())
            {
                std::sort(list.begin(), list.end(), [](const std::vector<float> &a, const std::vector<float> &b)
                          { return a[1] < b[1]; });
                same_id_filtered.push_back(static_cast<int>(list[0][0]));
            }
        };

        sort_and_append(list_1);
        sort_and_append(list_2);
        sort_and_append(list_3);

        return same_id_filtered;
    }

    Point2D RealEnvironment::calculate_wait_point(RobotHandle robot, Point2D target_point, float distance)
    {
        float angle = angle_by_two_point(robot->position(), target_point);
        if (-M_PI_2 - 0.01 <= angle && angle <= -M_PI_2 + 0.01) // Up wait point
            return Point2D(target_point.x(), target_point.y() + distance);
        if (M_PI_2 - 0.01 <= angle && angle <= M_PI_2 + 0.01) // Down wait point
            return Point2D(target_point.x(), target_point.y() - distance);
        if (M_PI - 0.01 <= fabs(normalize_angle(angle)) && fabs(normalize_angle(angle)) <= M_PI + 0.01) // Right wait point
            return Point2D(target_point.x() + distance, target_point.y());
        if (-0.01 <= angle && angle <= 0.01) // Left wait point
            return Point2D(target_point.x() - distance, target_point.y());
        return robot->position();
    }

    bool RealEnvironment::line_is_safe(RobotHandle robot, Point2D start_point, Point2D target_point)
    {
        for (int i = 0; i < swarm_.size(); i++)
        {
            if (swarm_[i]->id() == robot->id())
                continue;
            if (euclidean_distance(start_point,
                                   swarm_[i]->position()) < d_safe_junction_)
                return false;
            if (euclidean_distance(target_point,
                                   swarm_[i]->position()) < d_safe_junction_)
                return false;
            if (point_is_in_segment(start_point, target_point,
                                    swarm_[i]->position()))
                return false;
            if ((point_line_distance(start_point, target_point,
                                     swarm_[i]->position())) < d_safe_junction_)
                return false;
        }
        return true;
    }

    bool RealEnvironment::vertex_is_safe(RobotHandle robot, Point2D next_point)
    {
        for (int i = 0; i < swarm_.size(); i++)
        {
            if (swarm_[i]->id() == robot->id())
                continue;
            if (euclidean_distance(next_point,
                                   swarm_[i]->position()) < d_safe_junction_)
                return false;
        }
        return true;
    }

    bool RealEnvironment::compare_by_num_between(int id1, int id2)
    {
        return swarm_[id1]->num_between() < swarm_[id2]->num_between();
    }
    void RealEnvironment::visualize()
    {
        for (int i = 0; i < swarm_.size(); i++)
        {
            swarm_[i]->visualize(graph_);
        }
    }

    SwarmHandle RealEnvironment::swarm()
    {
        return swarm_;
    }

    GraphHandle RealEnvironment::graph_handle()
    {
        return graph_;
    }

    Graph RealEnvironment::graph()
    {
        return *graph_;
    }
    Point2D RealEnvironment::map_center() { return factory_->map_center(); }
    float RealEnvironment::map_length() { return factory_->map_length(); }
    float RealEnvironment::map_width() { return factory_->map_width(); }
    float RealEnvironment::single_line_width() { return factory_->single_line_width(); }
    float RealEnvironment::double_line_width() { return factory_->double_line_width(); }
}