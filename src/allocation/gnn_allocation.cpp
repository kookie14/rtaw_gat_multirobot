#include "allocation/gnn_allocation.h"

namespace rds
{
    GNNAllocation::GNNAllocation(SwarmHandle swarm, GraphHandle graph, TaskGeneratorHandle task_gen, GNNAllocationBufferHandle buffer,
                                 float map_length, float map_width, torch::DeviceType device, float dt, int mode)
    {
        swarm_ = swarm;
        robot_graph_ = graph;
        graph_ = std::make_shared<Graph>(*robot_graph_);
        task_gen_ = task_gen;
        buffer_ = buffer;
        map_length_ = map_length;
        map_width_ = map_width;
        dt_ = dt;
        mode_ = mode;
        max_path_cost_ = (map_length_ + map_width_);
        device_ = device;
        fixed_node_features_ = torch::zeros({(long)graph_->vertices().size(), 3});
        fixed_edge_features_ = torch::zeros({(long)graph_->edges().size(), 3});
        fixed_edge_index_ = torch::zeros({2, (long)graph_->edges().size()});
        calculate_fixed_node_features();
        calculate_fixed_edge_features();
    }

    void GNNAllocation::calculate_fixed_node_features()
    {
        for (Vertex v : graph_->vertices())
        {
            fixed_node_features_[v.id()][0] = 0.1f;
            fixed_node_features_[v.id()][1] = v.y() / map_width_;
            fixed_node_features_[v.id()][2] = v.x() / map_length_;
        }
    }

    void GNNAllocation::calculate_fixed_edge_features()
    {
        for (Edge e : graph_->edges())
        {
            fixed_edge_index_[0][e.id()] = e.start_id();
            fixed_edge_index_[1][e.id()] = e.end_id();

            fixed_edge_features_[e.id()][0] = e.direction() / M_PI;
            fixed_edge_features_[e.id()][1] = e.distance() / max_path_cost_;
            fixed_edge_features_[e.id()][2] = e.velocity();
        }
    }

    GNNAllocationObservation GNNAllocation::observation()
    {
        for (int i = 0; i < swarm_.size(); i++)
        {
            if (swarm_[i]->allocation_state() == FREE)
            {
                std::vector<torch::Tensor> obs = calculate_observation_ver2(swarm_[i]);
                return GNNAllocationObservation(true, i, obs[0], obs[1], obs[2], obs[3], obs[4], obs[5], obs[6]);
            }
            else
                continue;
        }
        return GNNAllocationObservation();
    }

    void GNNAllocation::allocation(int robot_id, int task_id)
    {
        if (mode_ == TRAINING)
        {
            TaskHandle task = task_gen_->task(task_id, robot_id, true);
            swarm_[robot_id]->set_task(*task);
            swarm_[robot_id]->set_reward_enabled(false);
            if (buffer_->is_full() == false)
            {
                swarm_[robot_id]->set_reward_enabled(true);
                swarm_[robot_id]->set_allocation_reward_idx(buffer_->counter());
            }
        }
        else if (mode_ == TESTING)
        {
            TaskHandle task = task_gen_->task(task_id, robot_id, true);
            swarm_[robot_id]->set_task(*task);
            swarm_[robot_id]->set_time_count_enabled(false);
            if (buffer_->is_full() == false)
            {
                swarm_[robot_id]->set_time_count_enabled(true);
                swarm_[robot_id]->set_time_count_idx(buffer_->counter());
            }
        }
    }

    void GNNAllocation::update_reward()
    {
        if (mode_ == TRAINING)
        {
            if (buffer_->is_complete() == true)
                return;
            for (int i = 0; i < swarm_.size(); i++)
            {
                if (swarm_[i]->reward_enabled() == true && swarm_[i]->done_count_reward() == true)
                {
                    if (buffer_->reward_states()[swarm_[i]->allocation_reward_idx()] == false)
                    {
                        float reward = round_num(-swarm_[i]->allocation_reward(), 2);
                        torch::Tensor reward_tensor = torch::tensor(reward, torch::TensorOptions(torch::kFloat32)).to(device_);
                        buffer_->update_reward(reward_tensor, swarm_[i]->allocation_reward_idx());
                        swarm_[i]->set_done_count_reward(false);
                    }
                }
            }
        }
        else if (mode_ = TESTING)
        {
            if (buffer_->is_complete() == true)
                return;
            for (int i = 0; i < swarm_.size(); i++)
            {
                if (swarm_[i]->time_count_enabled() == true && swarm_[i]->done_count_time() == true)
                {
                    if (buffer_->reward_states()[swarm_[i]->time_count_idx()] == false)
                    {
                        float reward = round_num(swarm_[i]->time_complete_task(), 2);
                        torch::Tensor reward_tensor = torch::tensor(reward, torch::TensorOptions(torch::kFloat32)).to(device_);
                        buffer_->update_reward(reward_tensor, swarm_[i]->time_count_idx());
                        swarm_[i]->set_done_count_time(false);
                    }
                }
            }
        }
    }

    std::vector<torch::Tensor> GNNAllocation::calculate_observation(RobotHandle selected)
    {
        graph_ = std::make_shared<Graph>(*robot_graph_);
        std::vector<TaskHandle> task_queue = task_gen_->task_queue();
        torch::Tensor node_features = torch::cat({fixed_node_features_, torch::zeros({int(swarm_.size()), 3})});
        for (int idx = 0; idx < task_queue.size(); idx++)
        {
            // Point in task queue
            node_features[task_queue[idx]->start_id()][0] = 0.2;

            node_features[task_queue[idx]->target_id()][0] = 0.2;

            // Route to start
            std::pair<std::vector<int>, std::vector<Point2D>> route_to_start = astar_planning(robot_graph_, selected->graph_id(), task_queue[idx]->start_id());
            std::pair<std::vector<int>, std::vector<Point2D>> route_to_target = astar_planning(robot_graph_, task_queue[idx]->start_id(), task_queue[idx]->target_id());

            std::vector<int> route_to_start_and_target(route_to_start.first);
            route_to_start_and_target.insert(route_to_start_and_target.end(), route_to_target.first.begin() + 1,
                                             route_to_target.first.end());

            std::vector<std::vector<float>> route_to_start_and_target_info = calculate_average_velocity_and_distance_forward(graph_, route_to_start_and_target);

            for (int i = 1; i < route_to_start_and_target.size(); i++)
            {
                graph_->add_edge(selected->graph_id(), route_to_start_and_target[i], route_to_start_and_target_info[0][i - 1],
                                 route_to_start_and_target_info[1][i - 1], route_to_start_and_target_info[2][i - 1]);
                graph_->add_edge(route_to_start_and_target[i], selected->graph_id(), route_to_start_and_target_info[0][i - 1],
                                 route_to_start_and_target_info[1][i - 1], route_to_start_and_target_info[2][i - 1]);
            }

            std::vector<std::vector<float>> route_to_target_info = calculate_average_velocity_and_distance_forward(graph_, route_to_target.first);

            // graph_->add_edge(task_queue[idx]->start_id(), task_queue[idx]->target_id(), route_to_target_info[0].back(),
            //                  route_to_target_info[1].back(), route_to_target_info[2].back());

            // std::pair<std::vector<int>, std::vector<Point2D>> route_to_start = astar_planning(robot_graph_, selected->graph_id(), task_queue[idx]->start_id());
            // std::pair<std::vector<int>, std::vector<Point2D>> route_to_target = astar_planning(robot_graph_, task_queue[idx]->start_id(), task_queue[idx]->target_id());

            // float route_to_start_vel = average_route_velocity(graph_, route_to_start.first);
            // float route_to_start_distance = calculate_path_cost(route_to_start.second);
            // float route_to_target_vel = average_route_velocity(graph_, route_to_target.first);
            // float route_to_target_distance = calculate_path_cost(route_to_target.second);

            // graph_->add_edge(selected->graph_id(), task_queue[idx]->start_id(), route_to_start_vel, route_to_start_distance);
            // graph_->add_edge(task_queue[idx]->start_id(), selected->graph_id(), route_to_start_vel, route_to_start_distance);

            // graph_->add_edge(selected->graph_id(), task_queue[idx]->target_id(), (route_to_start_vel + route_to_target_vel) / 2,
            //                  route_to_start_distance + route_to_target_distance);
            // graph_->add_edge(task_queue[idx]->target_id(), selected->graph_id(), (route_to_start_vel + route_to_target_vel) / 2,
            //                  route_to_start_distance + route_to_target_distance);

            // graph_->add_edge(task_queue[idx]->start_id(), task_queue[idx]->target_id(), route_to_target_vel, route_to_target_distance);
        }

        for (int id = 0; id < swarm_.size(); id++)
        {
            if (swarm_[id]->has_task())
            {
                // Robot assigned
                node_features[swarm_[id]->graph_id()][0] = 0.4;
                node_features[swarm_[id]->graph_id()][1] = swarm_[id]->x() / map_length_;
                node_features[swarm_[id]->graph_id()][2] = swarm_[id]->y() / map_width_;

                if (swarm_[id]->global_goal_type() == TO_START)
                {
                    // Point has occupied
                    node_features[swarm_[id]->start_id()][0] = 0.3;

                    node_features[swarm_[id]->target_id()][0] = 0.3;
                    //
                    std::pair<std::vector<int>, std::vector<Point2D>> route_to_start = astar_planning(robot_graph_, swarm_[id]->graph_id(), swarm_[id]->start_id());
                    std::pair<std::vector<int>, std::vector<Point2D>> route_to_target = astar_planning(robot_graph_, swarm_[id]->start_id(), swarm_[id]->target_id());

                    std::vector<int> route_to_start_and_target(route_to_start.first);
                    route_to_start_and_target.insert(route_to_start_and_target.end(), route_to_target.first.begin() + 1,
                                                     route_to_target.first.end());

                    std::vector<std::vector<float>> route_to_start_and_target_info = calculate_average_velocity_and_distance_forward(graph_, route_to_start_and_target);

                    for (int i = 1; i < route_to_start_and_target.size(); i++)
                    {
                        // graph_->add_edge(swarm_[id]->graph_id(), route_to_start_and_target[i], route_to_start_and_target_info[0][i - 1],
                        //                  route_to_start_and_target_info[1][i - 1], route_to_start_and_target_info[2][i - 1]);
                        graph_->add_edge(route_to_start_and_target[i], swarm_[id]->graph_id(), route_to_start_and_target_info[0][i - 1],
                                         route_to_start_and_target_info[1][i - 1], route_to_start_and_target_info[2][i - 1]);
                    }

                    std::vector<std::vector<float>> route_to_target_info = calculate_average_velocity_and_distance_forward(graph_, route_to_target.first);

                    graph_->add_edge(swarm_[id]->start_id(), swarm_[id]->target_id(), route_to_target_info[0].back(),
                                     route_to_target_info[1].back(), route_to_target_info[2].back());
                }
                else if (swarm_[id]->global_goal_type() == TO_TARGET)
                {
                    // Point has occupied
                    node_features[swarm_[id]->target_id()][0] = 0.3;

                    std::pair<std::vector<int>, std::vector<Point2D>> route_to_target = astar_planning(robot_graph_, swarm_[id]->graph_id(), swarm_[id]->target_id());
                    float route_to_target_vel = average_route_velocity(graph_, route_to_target.first);
                    float route_to_target_distance = calculate_path_cost(route_to_target.second);

                    std::vector<std::vector<float>> route_to_target_info = calculate_average_velocity_and_distance_forward(graph_, route_to_target.first);
                    for (int i = 1; i < route_to_target.first.size(); i++)
                    {
                        // graph_->add_edge(swarm_[id]->graph_id(), route_to_target.first[i], route_to_target_info[0][i - 1],
                        //                  route_to_target_info[1][i - 1], route_to_target_info[2][i - 1]);
                        graph_->add_edge(route_to_target.first[i], swarm_[id]->graph_id(), route_to_target_info[0][i - 1],
                                         route_to_target_info[1][i - 1], route_to_target_info[2][i - 1]);
                    }
                }
            }
            else if (swarm_[id]->has_task() == false)
            {
                if (swarm_[id]->id() == selected->id())
                {
                    node_features[swarm_[id]->graph_id()][0] = 0.6;
                    node_features[swarm_[id]->graph_id()][1] = swarm_[id]->x() / map_length_;
                    node_features[swarm_[id]->graph_id()][2] = swarm_[id]->y() / map_width_;
                }
                else
                {
                    node_features[swarm_[id]->graph_id()][0] = 0.5;
                    node_features[swarm_[id]->graph_id()][1] = swarm_[id]->x() / map_length_;
                    node_features[swarm_[id]->graph_id()][2] = swarm_[id]->y() / map_width_;
                }
            }

            for (int k = 0; k < swarm_.size(); k++)
            {
                if (k == id)
                    continue;
                // if (euclidean_distance(swarm_[id]->position(), swarm_[k]->position()) < 10)
                // {
                graph_->add_edge(swarm_[id]->graph_id(), swarm_[k]->graph_id(), 0.0, 0.0,
                                 calculate_difference_angle(swarm_[id]->theta(), swarm_[k]->theta()));
                // }
            }
        }
        int num_extra_edges = graph_->edges().size() - graph_->num_fixed_edges();
        torch::Tensor edge_features = torch::cat({fixed_edge_features_, torch::zeros({num_extra_edges, 3})});
        torch::Tensor edge_index = torch::cat({fixed_edge_index_, torch::zeros({2, num_extra_edges})}, 1);

        for (int idx = graph_->num_fixed_edges(); idx < graph_->edges().size(); idx++)
        {
            edge_index[0][idx] = graph_->edges()[idx].start_id();
            edge_index[1][idx] = graph_->edges()[idx].end_id();

            edge_features[idx][0] = graph_->edges()[idx].direction() / M_PI;
            edge_features[idx][1] = graph_->edges()[idx].distance() / max_path_cost_;
            edge_features[idx][2] = graph_->edges()[idx].velocity();
        }
        torch::Tensor start_ids = torch::zeros((int)task_queue.size());
        torch::Tensor target_ids = torch::zeros((int)task_queue.size());
        for (int i = 0; i < task_queue.size(); i++)
        {
            start_ids[i] = task_queue[i]->start_id();
            target_ids[i] = task_queue[i]->target_id();
        }
        torch::Tensor robot_ids = torch::arange(graph_->num_fixed_vertices(), (long)graph_->vertices().size()).reshape({1, -1});
        torch::Tensor selected_ids = torch::zeros({1, 1});
        selected_ids[0][0] = selected->graph_id();
        return {node_features.toType(torch::kFloat32).to(device_),
                edge_features.toType(torch::kFloat32).to(device_),
                edge_index.toType(torch::kInt64).to(device_),
                start_ids.toType(torch::kInt64).to(device_),
                target_ids.toType(torch::kInt64).to(device_),
                robot_ids.toType(torch::kInt64).to(device_),
                selected_ids.toType(torch::kInt64).to(device_)};
    }

    std::vector<torch::Tensor> GNNAllocation::calculate_observation_ver2(RobotHandle selected)
    {
        graph_ = std::make_shared<Graph>(*robot_graph_);
        std::vector<TaskHandle> task_queue = task_gen_->task_queue();
        torch::Tensor node_features = torch::cat({fixed_node_features_, torch::zeros({int(swarm_.size()), 3})});
        for (int idx = 0; idx < task_queue.size(); idx++)
        {
            // Point in task queue
            node_features[task_queue[idx]->start_id()][0] = 0.2;

            node_features[task_queue[idx]->target_id()][0] = 0.3;

            // Route to start
            std::pair<std::vector<int>, std::vector<Point2D>> route_to_start = astar_planning(robot_graph_, selected->graph_id(), task_queue[idx]->start_id());
            std::pair<std::vector<int>, std::vector<Point2D>> route_to_target = astar_planning(robot_graph_, task_queue[idx]->start_id(), task_queue[idx]->target_id());

            std::vector<std::vector<float>> route_to_start_info = calculate_average_velocity_and_distance_backward(graph_, route_to_start.first);
            std::vector<std::vector<float>> route_to_target_info = calculate_average_velocity_and_distance_backward(graph_, route_to_target.first);
            for (int i = 0; i < route_to_start.first.size() - 1; i++)
            {
                graph_->add_edge(route_to_start.first[i], task_queue[idx]->start_id(), route_to_start_info[0][i],
                                 route_to_start_info[1][i], route_to_start_info[2][i]);
            }

            for (int i = 0; i < route_to_target.first.size() - 1; i++)
            {
                graph_->add_edge(route_to_target.first[i], task_queue[idx]->target_id(), route_to_target_info[0][i],
                                 route_to_target_info[1][i], route_to_target_info[2][i]);
            }

            graph_->add_edge(selected->graph_id(), task_queue[idx]->target_id(), (route_to_start_info[0].front() + route_to_target_info[0].front()) / 2,
                             route_to_start_info[1].front() + route_to_target_info[1].front());
        }

        for (int id = 0; id < swarm_.size(); id++)
        {
            if (swarm_[id]->has_task())
            {
                // Robot assigned
                node_features[swarm_[id]->graph_id()][0] = 0.6;
                node_features[swarm_[id]->graph_id()][1] = swarm_[id]->x() / map_length_;
                node_features[swarm_[id]->graph_id()][2] = swarm_[id]->y() / map_width_;

                if (swarm_[id]->global_goal_type() == TO_START)
                {
                    // Point has occupied
                    node_features[swarm_[id]->start_id()][0] = 0.4;

                    node_features[swarm_[id]->target_id()][0] = 0.5;
                    //
                    std::pair<std::vector<int>, std::vector<Point2D>> route_to_start = astar_planning(robot_graph_, swarm_[id]->graph_id(), swarm_[id]->start_id());
                    std::pair<std::vector<int>, std::vector<Point2D>> route_to_target = astar_planning(robot_graph_, swarm_[id]->start_id(), swarm_[id]->target_id());

                    std::vector<int> route_to_start_and_target(route_to_start.first);
                    route_to_start_and_target.insert(route_to_start_and_target.end(), route_to_target.first.begin() + 1,
                                                     route_to_target.first.end());

                    std::vector<std::vector<float>> route_to_start_and_target_info = calculate_average_velocity_and_distance_forward(graph_, route_to_start_and_target);

                    for (int i = 1; i < route_to_start_and_target.size(); i++)
                    {
                        graph_->add_edge(route_to_start_and_target[i], swarm_[id]->graph_id(), route_to_start_and_target_info[0][i - 1],
                                         route_to_start_and_target_info[1][i - 1], route_to_start_and_target_info[2][i - 1]);
                    }
                }
                else if (swarm_[id]->global_goal_type() == TO_TARGET)
                {
                    // Point has occupied
                    node_features[swarm_[id]->target_id()][0] = 0.5;

                    std::pair<std::vector<int>, std::vector<Point2D>> route_to_target = astar_planning(robot_graph_, swarm_[id]->graph_id(), swarm_[id]->target_id());

                    std::vector<std::vector<float>> route_to_target_info = calculate_average_velocity_and_distance_forward(graph_, route_to_target.first);
                    for (int i = 1; i < route_to_target.first.size(); i++)
                    {
                        graph_->add_edge(route_to_target.first[i], swarm_[id]->graph_id(), route_to_target_info[0][i - 1],
                                         route_to_target_info[1][i - 1], route_to_target_info[2][i - 1]);
                    }
                }
            }
            else if (swarm_[id]->has_task() == false)
            {
                if (swarm_[id]->id() == selected->id())
                {
                    node_features[swarm_[id]->graph_id()][0] = 0.8;
                    node_features[swarm_[id]->graph_id()][1] = swarm_[id]->x() / map_length_;
                    node_features[swarm_[id]->graph_id()][2] = swarm_[id]->y() / map_width_;
                    for (int k = 0; k < swarm_.size(); k++)
                    {
                        if (swarm_[k]->id() == selected->id()) continue;
                        graph_->add_edge(swarm_[k]->graph_id(), selected->graph_id(), 0.0, 0.0,
                                        calculate_difference_angle(swarm_[id]->theta(), swarm_[k]->theta()));
                    }
                }
                else
                {
                    node_features[swarm_[id]->graph_id()][0] = 0.7;
                    node_features[swarm_[id]->graph_id()][1] = swarm_[id]->x() / map_length_;
                    node_features[swarm_[id]->graph_id()][2] = swarm_[id]->y() / map_width_;
                }
            }

            // for (int k = 0; k < swarm_.size(); k++)
            // {
            //     if (k == id)
            //         continue;
            //     // if (euclidean_distance(swarm_[id]->position(), swarm_[k]->position()) < 10)
            //     // {
            //         graph_->add_edge(swarm_[id]->graph_id(), swarm_[k]->graph_id(), 0.0, 0.0,
            //                         calculate_difference_angle(swarm_[id]->theta(), swarm_[k]->theta()));
            //     // }
            // }
        }
        int num_extra_edges = graph_->edges().size() - graph_->num_fixed_edges();
        torch::Tensor edge_features = torch::cat({fixed_edge_features_, torch::zeros({num_extra_edges, 3})});
        torch::Tensor edge_index = torch::cat({fixed_edge_index_, torch::zeros({2, num_extra_edges})}, 1);

        for (int idx = graph_->num_fixed_edges(); idx < graph_->edges().size(); idx++)
        {
            edge_index[0][idx] = graph_->edges()[idx].start_id();
            edge_index[1][idx] = graph_->edges()[idx].end_id();

            edge_features[idx][0] = graph_->edges()[idx].direction() / M_PI;
            edge_features[idx][1] = graph_->edges()[idx].distance() / max_path_cost_;
            edge_features[idx][2] = graph_->edges()[idx].velocity();
        }
        torch::Tensor start_ids = torch::zeros((int)task_queue.size());
        torch::Tensor target_ids = torch::zeros((int)task_queue.size());
        for (int i = 0; i < task_queue.size(); i++)
        {
            start_ids[i] = task_queue[i]->start_id();
            target_ids[i] = task_queue[i]->target_id();
        }
        torch::Tensor robot_ids = torch::arange(graph_->num_fixed_vertices(), (long)graph_->vertices().size()).reshape({1, -1});
        torch::Tensor selected_ids = torch::zeros({1, 1});
        selected_ids[0][0] = selected->graph_id();
        return {node_features.toType(torch::kFloat32).to(device_),
                edge_features.toType(torch::kFloat32).to(device_),
                edge_index.toType(torch::kInt64).to(device_),
                start_ids.toType(torch::kInt64).to(device_),
                target_ids.toType(torch::kInt64).to(device_),
                robot_ids.toType(torch::kInt64).to(device_),
                selected_ids.toType(torch::kInt64).to(device_)};
    }
}