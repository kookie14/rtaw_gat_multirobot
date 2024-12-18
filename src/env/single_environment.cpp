#include "env/single_environment.h"

namespace rds
{
    SingleEnvironment::SingleEnvironment(int initial_mode, std::string data_path, int num_robots, float robot_max_speed,
                                         float robot_max_payload, matplot::axes_handle visual, int num_zones_in_row,
                                         int num_zones_in_col, float waiting_time)
    {
        dt_ = 0.1;
        data_path_ = data_path;
        factory = std::make_shared<SingleReconstruction>(data_path);
        factory->visualize(visual, true);
        graph_ = std::make_shared<Graph>(factory->graph());
        num_zones_in_col_ = num_zones_in_col;
        num_zones_in_row_ = num_zones_in_row;
        calculate_graph_zone();
        num_robots_ = num_robots;
        if (initial_mode == RANDOM_INITIALIZATION)
            random_initialization(robot_max_speed, robot_max_payload, visual, waiting_time);
        else if (initial_mode == TEST_INITIALIZATION)
            test_initialization(robot_max_speed, robot_max_payload, visual, waiting_time);
        else
            waiting_initialization(robot_max_speed, robot_max_payload, visual, waiting_time);
        d_safe_junction_ = swarm_[0]->length() + 0.01;
        d_safe_closest_ = swarm_[0]->length() + 0.2;
    }

    void SingleEnvironment::waiting_initialization(float robot_max_speed, float robot_max_payload,
                                                   matplot::axes_handle visual, float waiting_time)
    {
        Vertices homes = factory->waiting_vertices();
        assert(homes.size() >= num_robots_);
        int index = 0;
        for (int id = 0; id < num_robots_; id++)
        {
            int home_id = index;
            if (id % 2 == 1)
            {
                home_id = index + int(homes.size() / 2);
                index += 1;
            }
            if (homes[id].y() > factory->map_width() / 2)
            {
                swarm_.push_back(std::make_shared<Robot>(id, dt_, Pose2D(homes[home_id].x(), homes[home_id].y(), -M_PI_2),
                                                         robot_max_speed, robot_max_payload, visual, 0.98, 0.61));
                swarm_[id]->set_waiting_time(waiting_time);
            }
            else
            {
                swarm_.push_back(std::make_shared<Robot>(id, dt_, Pose2D(homes[home_id].x(), homes[home_id].y(), M_PI_2),
                                                         robot_max_speed, robot_max_payload, visual, 0.98, 0.61));
                swarm_[id]->set_waiting_time(waiting_time);
            }
            initial_poses_.push_back(swarm_[id]->pose());
        }
    }

    void SingleEnvironment::random_initialization(float robot_max_speed, float robot_max_payload,
                                                  matplot::axes_handle visual, float waiting_time)
    {
        Vertices vertices;
        for (Vertex v : graph_->vertices())
        {
            if (v.type() == WORKING_VERTEX || v.type() == STORAGE_VERTEX || v.type() == CHARGING_VERTEX || v.type() == WAITING_VERTEX)
            {
                vertices.push_back(v);
            }
        }
        Vertex init_vertex;
        int idx;
        float angle;
        for (int id = 0; id < num_robots_; id++)
        {
            idx = torch::randint(0, vertices.size() - 1, 1).item().toInt();
            init_vertex = vertices[idx];
            vertices.erase(vertices.begin() + idx);
            if (init_vertex.type() == WORKING_VERTEX)
            {
                angle = angle_by_two_point(init_vertex.position(), graph_->vertex(init_vertex.neighbors()[0]).position());
                swarm_.push_back(std::make_shared<Robot>(id, dt_, Pose2D(init_vertex.x(), init_vertex.y(), angle),
                                                         robot_max_speed, robot_max_payload, visual, 0.98, 0.61));
            }
            else
            {
                swarm_.push_back(std::make_shared<Robot>(id, dt_, Pose2D(init_vertex.x(), init_vertex.y(), M_PI_2),
                                                         robot_max_speed, robot_max_payload, visual, 0.98, 0.61));
            }
            swarm_[id]->set_waiting_time(waiting_time);
            initial_poses_.push_back(swarm_[id]->pose());
        }
    }

    void SingleEnvironment::test_initialization(float robot_max_speed, float robot_max_payload,
                                                matplot::axes_handle visual, float waiting_time)
    {
        std::vector<std::vector<float>> init_vertex_ids = load_txt(data_path_ + "/init_vertex_ids.txt");
        Vertex init_vertex;
        float angle;
        for (int id = 0; id < num_robots_; id++)
        {
            init_vertex = graph_->vertex(int(init_vertex_ids[0][id]));
            if (init_vertex.type() == WORKING_VERTEX)
            {
                angle = angle_by_two_point(init_vertex.position(), graph_->vertex(init_vertex.neighbors()[0]).position());
                swarm_.push_back(std::make_shared<Robot>(id, dt_, Pose2D(init_vertex.x(), init_vertex.y(), angle),
                                                         robot_max_speed, robot_max_payload, visual, 0.98, 0.61));
            }
            else
            {
                swarm_.push_back(std::make_shared<Robot>(id, dt_, Pose2D(init_vertex.x(), init_vertex.y(), M_PI_2),
                                                         robot_max_speed, robot_max_payload, visual, 0.98, 0.61));
            }
            swarm_[id]->set_waiting_time(waiting_time);
            initial_poses_.push_back(swarm_[id]->pose());
        }
    }

    void SingleEnvironment::reset()
    {
        for (int id = 0; id < swarm_.size(); id++)
        {
            swarm_[id]->set_pose(initial_poses_[id]);
            swarm_[id]->set_allocation_state(FREE);
            swarm_[id]->clear_task();
            swarm_[id]->clear_route();
            swarm_[id]->set_done_count_reward(false);
            swarm_[id]->set_done_count_time(false);
        }
    }

    void SingleEnvironment::add_all_robot_graph()
    {
        graph_->delete_robot_vertices();
        for (RobotHandle robot : swarm_)
        {
            add_robot_to_graph(robot);

            if (graph_->neighbors(robot->graph_id()).size() == 0)
            {
                for (int i = 0; i < graph_zones_.size(); i++)
                {
                    bool cond1 = graph_zones_[i]->x() - graph_zones_[i]->length() / 2 <= robot->x() && graph_zones_[i]->x() + graph_zones_[i]->length() / 2 >= robot->x();
                    bool cond2 = graph_zones_[i]->y() - graph_zones_[i]->width() / 2 <= robot->y() && graph_zones_[i]->y() + graph_zones_[i]->width() / 2 >= robot->y();
                    if (cond1 && cond2)
                    {
                        float min_distance = 100000;
                        float distance;
                        int neighbor;
                        for (Vertex v: graph_zones_[i]->vertices())
                        {
                            // if (fabs(calculate_difference_angle(angle_by_two_point(robot->position(), v.position()), robot->theta())) < 0.2)
                            // {
                                distance = euclidean_distance(v.position(), robot->position());
                                if (distance < min_distance)   
                                {
                                    min_distance = distance;   
                                    neighbor = v.id();
                                }
                            // }
                        }
                        graph_->add_edge(robot->graph_id(), neighbor);
                        break;
                    }
                }

                // std::cout << "Neighbor of robot " << robot->id() << ": "<< graph_->neighbors(robot->graph_id()) << std::endl;
            }
        }
    }

    void SingleEnvironment::add_robot_to_graph(RobotHandle robot)
    {
        graph_->add_vertex(ROBOT_VERTEX, robot->position());
        robot->set_graph_id(graph_->vertex(-1).id());
        bool cond1, cond2;
        for (int i = 0; i < graph_zones_.size(); i++)
        {
            cond1 = graph_zones_[i]->x() - graph_zones_[i]->length() / 2 <= robot->x() && graph_zones_[i]->x() + graph_zones_[i]->length() / 2 >= robot->x();
            cond2 = graph_zones_[i]->y() - graph_zones_[i]->width() / 2 <= robot->y() && graph_zones_[i]->y() + graph_zones_[i]->width() / 2 >= robot->y();
            if (cond1 && cond2)
            {
                find_neighbor_of_robot_in_zone(robot, graph_zones_[i]);
                break;
            }
        }
        // std::cout << "Neighbor of robot " << robot->id() << ": ";
        // for (int neighbor: graph_->neighbors(robot->graph_id())) std::cout << neighbor << " ";
        // std::cout << std::endl;
    }

    std::vector<GraphZoneHandle> SingleEnvironment::find_neighbor_of_zone(GraphZoneHandle zone)
    {
        std::vector<GraphZoneHandle> zone_list = {zone};
        std::vector<std::vector<int>> row_col_ids = {{zone->row_id() - 1, zone->col_id()},
                                                     {zone->row_id(), zone->col_id() - 1},
                                                     {zone->row_id(), zone->col_id() + 1},
                                                     {zone->row_id() + 1, zone->col_id()}};
        for (std::vector<int> ids : row_col_ids)
        {
            if (ids[0] >= 0 && ids[0] < num_zones_in_row_ && ids[1] >= 0 && ids[1] < num_zones_in_col_)
            {
                zone_list.push_back(graph_zones_[ids[0] * num_zones_in_col_ + ids[1]]);
            }
        }
        return zone_list;
    }

    void SingleEnvironment::find_neighbor_of_robot_in_zone(RobotHandle robot, GraphZoneHandle zone)
    {
        float dx1, dy1, dx2, dy2, distance;
        for (Vertex v : zone->vertices())
        {
            dx1 = graph_->vertex(v.id()).x() - robot->x();
            dy1 = graph_->vertex(v.id()).y() - robot->y();

            if (fabs(dx1) < SAME_DIST && fabs(dy1) < SAME_DIST)
            {
                for (int neighbor : graph_->neighbors(v.id()))
                {
                    // if (robot->id() == 0) std::cout << neighbor << std::endl;

                    graph_->add_edge(robot->graph_id(), neighbor, graph_->edge(v.id(), neighbor).velocity());
                }
                return;
            }
            else
            {
                for (int neighbor : graph_->neighbors(v.id()))
                {
                    distance = point_line_distance(v.position(), graph_->vertex(neighbor).position(), robot->position());
                    if (distance <= factory->single_line_width() || distance <= factory->double_line_width() / 2)
                    {
                        dx2 = graph_->vertex(neighbor).x() - robot->x();
                        dy2 = graph_->vertex(neighbor).y() - robot->y();
                        if ((dx1 * dx2 < 0 && dy1 * dy2 >= 0) || (dx1 * dx2 >= 0 && dy1 * dy2 < 0))
                        {
                            if (graph_->neighbors(robot->graph_id()).size() > 0)
                            {
                                if (number_exists(graph_->neighbors(robot->graph_id()), neighbor))
                                    continue;
                            }
                            graph_->add_edge(robot->graph_id(), neighbor, graph_->edge(v.id(), neighbor).velocity());
                        }
                    }
                }
            }
        }
    }

    void SingleEnvironment::calculate_graph_zone()
    {
        if (num_zones_in_col_ == 1 && num_zones_in_row_ == 1)
        {
            graph_zones_.push_back(std::make_shared<GraphZone>(1, 1, 0, factory->map_center(), factory->map_length(), factory->map_width()));
            for (Vertex v : graph_->vertices())
            {
                graph_zones_[0]->add_vertex(v);
            }
        }
        else
        {
            float zone_length = round_num(factory->map_length() / num_zones_in_col_);
            float zone_width = round_num(factory->map_width() / num_zones_in_row_);
            float start_x = factory->map_center().x() - factory->map_length() / 2 + zone_length / 2;
            float start_y = factory->map_center().y() - factory->map_width() / 2 + zone_width / 2;
            float center_x, center_y;
            for (int col = 0; col < num_zones_in_col_; col++)
            {
                center_x = start_x + col * zone_length;
                for (int row = 0; row < num_zones_in_row_; row++)
                {
                    center_y = start_y + row * zone_width;
                    graph_zones_.push_back(std::make_shared<GraphZone>(row, col, row + col * num_zones_in_row_,
                                                                       Point2D(center_x, center_y), zone_length, zone_width));
                }
            }
            for (Vertex v : graph_->vertices())
            {
                for (int i = 0; i < graph_zones_.size(); i++)
                {
                    if (graph_zones_[i]->add_vertex(v))
                        break;
                }
            }
        }
    }

    void SingleEnvironment::controller(float wait_time)
    {
        std::vector<bool> traffic_control;
        for (int i = 0; i < swarm_.size(); i++)
        {
            swarm_[i]->set_waiting_time(wait_time);
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
        std::sort(id_list.begin(), id_list.end(), [this](int a, int b)
                  { return swarm_[a]->num_between() < swarm_[b]->num_between(); });
        for (int id : id_list)
        {
            if (traffic_control[id] == true)
            {
                traffic_assign_control_state(swarm_[id]);
            }
        }
    }

    bool SingleEnvironment::task_assign_control_state(RobotHandle robot, float wait_time)
    {
        if (robot->has_task() == false)
        {
            return true;
        }
        else
        {
            if (robot->has_route() == false)
            {
                return true;
            }
            else
            {
                if (robot->route_done() == false)
                {
                    if (robot->start_reward_count())
                    {
                        robot->plus_allocation_reward();
                    }
                    if (robot->start_count_time())
                    {
                        robot->plus_time_completed_task();
                    }
                    if (robot->route_type() == TO_START)
                    {
                        robot->set_allocation_state(ON_WAY_TO_START);
                    }
                    if (robot->route_type() == TO_TARGET)
                    {
                        robot->set_allocation_state(ON_WAY_TO_TARGET);
                    }
                    if (robot->route_type() == TO_WAITING)
                    {
                        robot->set_allocation_state(ON_WAY_TO_WAITING);
                    }
                    if (robot->route_type() == TO_CHARGING)
                    {
                        robot->set_allocation_state(ON_WAY_TO_CHARGING);
                    }
                    return false;
                }
                else
                {
                    if (robot->wait_for_picking_or_dropping(wait_time) == false)
                    {
                        if (robot->start_count_time())
                        {
                            robot->plus_time_completed_task();
                        }
                        return true;
                    }
                    else
                    {
                        if (robot->route_type() == TO_START)
                        {
                            robot->set_start_reward_count(false);
                            robot->set_done_count_reward(true);
                            robot->start_is_done();
                            robot->clear_route();
                            return true;
                        }
                        else if (robot->route_type() == TO_TARGET)
                        {
                            robot->set_start_count_time(false);
                            robot->set_done_count_time(true);
                            robot->set_allocation_state(FREE);
                            robot->clear_route();
                            robot->clear_task();
                            robot->set_task_done(true);
                            return true;
                        }
                        else
                        {
                            robot->clear_route();
                            return true;
                        }
                    }
                }
            }
        }
    }

    void SingleEnvironment::traffic_assign_control_state(RobotHandle robot)
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
                // if (point_is_in_segment(robot->position(), closest_robot->position(), robot->next_point_in_route()))
                // {
                //     robot->set_control_next_point(robot->next_point_in_route());
                //     robot->set_control_state(TO_NEXT_POINT);
                // }
                // else
                // {
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
                // }
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
        if (robot->next_point_id() == 1)
            robot->run();
        else
        {
            robot->run(graph_->edge(robot->route_graph_id(robot->next_point_id() - 1),
                                    robot->route_graph_id(robot->next_point_id()))
                           .velocity());
        }
    }

    void SingleEnvironment::set_control_state_to_pass_same_point(RobotHandle robot)
    {
        if (euclidean_distance(robot->position(), robot->same_point()) < d_safe_junction_)
        {
            robot->set_control_next_point(robot->same_point());
            robot->set_control_state(TO_NEXT_POINT);
        }
        else
        {
            std::vector<float> wait_time_list = robot->wait_time_list();
            std::vector<float> dist_to_same_list = robot->dist_to_same_list();
            auto max_wait_time = *std::max_element(wait_time_list.begin(), wait_time_list.end());
            auto min_dist = *std::min_element(dist_to_same_list.begin(), dist_to_same_list.end());

            auto max_wait_time_index = std::distance(wait_time_list.begin(), std::max_element(wait_time_list.begin(), wait_time_list.end()));
            auto min_dist_index = std::distance(dist_to_same_list.begin(), std::min_element(dist_to_same_list.begin(), dist_to_same_list.end()));

            if (max_wait_time_index == 0)
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
                    if (min_dist_index == 0)
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

    void SingleEnvironment::calculate_same_and_between(RobotHandle robot)
    {
        std::vector<int> same_line_list = {};
        std::vector<int> same_vertex_list;
        float min_same_dist = std::numeric_limits<float>::infinity();
        bool angle_cond, cond1, cond2, cond3, cond4;
        float angle1, angle2;
        for (const auto other : swarm_)
        {
            if (other->id() == robot->id() || other->has_task() == false || other->has_route() == false)
                continue;
            std::pair<bool, Point2D> intersect = check_segment_intersection(robot->position(), robot->next_point_in_route(),
                                                                            other->position(), other->next_point_in_route());
            angle1 = angle_by_two_point(other->position(), other->next_point_in_route());
            angle2 = angle_by_two_point(robot->position(), robot->next_point_in_route());
            angle_cond = fabs(calculate_difference_angle(angle1, angle2)) < 0.1;
            if (intersect.first)
            {
                cond1 = min_same_dist > euclidean_distance(robot->position(), intersect.second) && euclidean_distance(robot->position(), intersect.second) > 0;
                cond2 = point_is_in_segment(robot->position(), robot->next_point_in_route(), intersect.second);
                cond3 = point_is_in_segment(other->position(), other->next_point_in_route(), intersect.second);
                if (cond1 && cond2 && cond3 && angle_cond == false)
                {
                    min_same_dist = euclidean_distance(robot->position(), intersect.second);
                    robot->set_same_point(intersect.second);
                    same_vertex_list.push_back(other->id());
                }
            }
            if (angle_cond && point_is_in_segment(robot->position(), robot->next_point_in_route(), other->position()))
            {
                same_line_list.push_back(other->id());
            }
        }
        robot->set_num_between(same_line_list.size());
        if (same_line_list.size() == 1)
            robot->set_closest_robot_id(same_line_list[0]);
        else if (same_line_list.size() > 1)
        {
            float min_distance = euclidean_distance(robot->position(), robot->next_point_in_route());
            float distance;
            for (int i = 0; i < same_line_list.size(); i++)
            {
                distance = euclidean_distance(robot->position(), swarm_[same_line_list[i]]->position());
                if (distance < min_distance)
                {
                    min_distance = distance;
                    robot->set_closest_robot_id(same_line_list[i]);
                }
            }
        }

        if (!same_vertex_list.empty())
        {
            std::vector<float> wait_time_list = {robot->wait_time_for_priority()};
            std::vector<float> dist_to_same_list = {euclidean_distance(robot->position(), robot->same_point())};
            std::vector<int> same_id_filtered = filter_same_vertex_list(robot, same_vertex_list);
            for (int id : same_id_filtered)
            {
                wait_time_list.push_back(swarm_[id]->wait_time_for_priority());
                dist_to_same_list.push_back(
                    euclidean_distance(swarm_[id]->position(), robot->same_point()));
            }
            robot->set_same_next_ids(same_id_filtered);
            robot->set_num_same_next(same_id_filtered.size());
            robot->set_wait_time_list(wait_time_list);
            robot->set_dist_to_same_list(dist_to_same_list);
        }
        else
        {
            robot->set_num_same_next(same_vertex_list.size());
            robot->set_same_next_ids(same_vertex_list);
        }
    }

    std::vector<int> SingleEnvironment::filter_same_vertex_list(RobotHandle robot, std::vector<int> same_vertex_list)
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
                if (list_1.empty() || std::fabs(calculate_difference_angle(data[2], list_1[0][2])) < 0.1)
                {
                    list_1.push_back(data);
                }
                else if (list_2.empty() || std::fabs(calculate_difference_angle(data[2], list_2[0][2])) < 0.1)
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

    Point2D SingleEnvironment::calculate_wait_point(RobotHandle robot, Point2D target_point, float distance)
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

    bool SingleEnvironment::line_is_safe(RobotHandle robot, Point2D start_point, Point2D target_point)
    {
        for (int i = 0; i < swarm_.size(); i++)
        {
            if (swarm_[i]->id() == robot->id())
                continue;
            if (euclidean_distance(start_point, swarm_[i]->position()) < d_safe_junction_)
                return false;
            if (euclidean_distance(target_point, swarm_[i]->position()) < d_safe_junction_)
                return false;
            if (point_is_in_segment(start_point, target_point,
                                    swarm_[i]->position()))
                return false;
            if ((point_line_distance(start_point, target_point, swarm_[i]->position())) < d_safe_junction_)
                return false;
        }
        return true;
    }

    bool SingleEnvironment::vertex_is_safe(RobotHandle robot, Point2D next_point)
    {
        for (int i = 0; i < swarm_.size(); i++)
        {
            if (swarm_[i]->id() == robot->id())
                continue;
            if (euclidean_distance(next_point, swarm_[i]->position()) < d_safe_junction_)
                return false;
        }
        return true;
    }

    bool SingleEnvironment::compare_by_num_between(int id1, int id2)
    {
        return swarm_[id1]->num_between() < swarm_[id2]->num_between();
    }
    void SingleEnvironment::visualize()
    {
        for (int i = 0; i < swarm_.size(); i++)
        {
            swarm_[i]->visualize(graph_);
        }
    }

    SwarmHandle SingleEnvironment::swarm()
    {
        return swarm_;
    }

    GraphHandle SingleEnvironment::graph_handle()
    {
        return graph_;
    }

    Graph SingleEnvironment::graph()
    {
        return *graph_;
    }
    Point2D SingleEnvironment::map_center() { return factory->map_center(); }
    float SingleEnvironment::map_length() { return factory->map_length(); }
    float SingleEnvironment::map_width() { return factory->map_width(); }
    float SingleEnvironment::single_line_width() { return factory->single_line_width(); }
    float SingleEnvironment::double_line_width() { return factory->double_line_width(); }
}