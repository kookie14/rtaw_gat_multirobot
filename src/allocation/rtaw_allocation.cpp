#include "allocation/rtaw_allocation.h"

namespace rds
{
    RTAWAllocation::RTAWAllocation(SwarmHandle swarm, GraphHandle graph, TaskGeneratorHandle task_gen, RTAWBufferHandle buffer,
                                   float map_length, float map_width, torch::DeviceType device, float dt, int mode)
    {
        swarm_ = swarm;
        graph_ = graph;
        task_gen_ = task_gen;
        buffer_ = buffer;
        map_length_ = map_length;
        map_width_ = map_width;
        dt_ = dt;
        mode_ = mode;
        max_path_cost_ = 2 * (map_length_ + map_width_);
        max_rest_time_ = 2 * (map_length_ + map_width_) * dt_;
        device_ = device;
    }

    RTAWObservation RTAWAllocation::observation()
    {
        for (int i = 0; i < swarm_.size(); i++)
        {
            if (swarm_[i]->allocation_state() == FREE)
            {
                std::vector<torch::Tensor> obs = calculate_observation(swarm_[i]);
                return RTAWObservation(true, i, obs[0], obs[1], obs[2]);
            }
            else continue;  
        }
        return RTAWObservation();
    }

    void RTAWAllocation::allocation(int robot_id, int task_id)
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

    void RTAWAllocation::update_reward()
    {
        if (mode_ == TRAINING)
        {
            if (buffer_->is_complete() == true) return;
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
            if (buffer_->is_complete() == true) return;
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

    std::vector<torch::Tensor> RTAWAllocation::calculate_observation(RobotHandle selected)
    {
        torch::Tensor selected_data = torch::zeros({1, 3});
        selected_data[0][0] = float(selected->x()/map_length_);
        selected_data[0][1] = float(selected->y()/map_width_);
        selected_data[0][2] = 0.0;
        return {selected_data.toType(torch::kFloat32).to(device_), calculate_robot_observation(), calculate_task_observation(selected)};
    }

    torch::Tensor RTAWAllocation::calculate_task_observation(RobotHandle selected)
    {
        std::vector<TaskHandle> task_queue = task_gen_->task_queue();
        torch::Tensor task_out = torch::zeros({1, (int)task_queue.size(), 6});
        for (int i = 0; i < task_queue.size(); i++)
        {
            task_out[0][i][0] = float(task_queue[i]->start_x()/map_length_);
            task_out[0][i][1] = float(task_queue[i]->start_y()/map_width_);
            task_out[0][i][2] = float(task_queue[i]->target_x()/map_length_);
            task_out[0][i][3] = float(task_queue[i]->target_y()/map_width_);
            task_out[0][i][4] = float(astar_planning_cost(graph_, selected->graph_id(), task_queue[i]->start_id())/max_path_cost_);
            task_out[0][i][5] = float(astar_planning_cost(graph_, task_queue[i]->start_id(), task_queue[i]->target_id())/max_path_cost_);
        }

        return task_out.toType(torch::kFloat32).to(device_);
    }
    torch::Tensor RTAWAllocation::calculate_robot_observation()
    {
        torch::Tensor robot_out = torch::zeros({1, (int)swarm_.size(), 3});
        for (int i = 0; i < swarm_.size(); i++)
        {
            robot_out[0][i][0] = float(swarm_[i]->x()/map_length_);
            robot_out[0][i][1] = float(swarm_[i]->y()/map_width_);
            robot_out[0][i][2] = float(estimate_rest_time(swarm_[i])/max_rest_time_);
        }
        return robot_out.toType(torch::kFloat32).to(device_);
    }

    float RTAWAllocation::estimate_rest_time(RobotHandle robot)
    {
        if (robot->has_task() == false && robot->has_route() == false)
        {
            return 0.0f;
        }
        else if (robot->has_task() == true && robot->has_route() == false)
        {
            if (robot->global_goal_type() == TO_START)
            {
                float rest_time = astar_planning_cost(graph_, robot->graph_id(), robot->start_id()) * dt_ + astar_planning_cost(graph_, robot->start_id(), robot->target_id()) * dt_ + 2 * robot->waiting_time();
                return rest_time * (1 + torch::normal(0.25f, 0.5f, 1).clamp(0.0f, 0.5f).item().toFloat());
            }
            else if (robot->global_goal_type() == TO_TARGET)
            {
                float rest_time = astar_planning_cost(graph_, robot->graph_id(), robot->target_id()) * dt_ + robot->waiting_time();
                return rest_time * (1 + torch::normal(0.25f, 0.5f, 1).clamp(0.0f, 0.5f).item().toFloat());
            }
            else
                return 0.0;
        }
        else
        {
            if (robot->route_type() == TO_START)
            {
                float rest_time;
                if (is_same_point(robot->position(), robot->global_goal_position()))
                {
                    rest_time = robot->waiting_time() - robot->wait_time_by_picking() + robot->waiting_time() + astar_planning_cost(graph_, robot->graph_id(), robot->target_id()) * dt_;
                }
                else
                {
                    rest_time = astar_planning_cost(graph_, robot->graph_id(), robot->start_id()) * dt_ + astar_planning_cost(graph_, robot->start_id(), robot->target_id()) * dt_ + 2 * robot->waiting_time();
                }
                return rest_time * (1 + torch::normal(0.25f, 0.5f, 1).clamp(0.0f, 0.5f).item().toFloat());
            }
            else if (robot->route_type() == TO_TARGET)
            {
                float rest_time;
                if (is_same_point(robot->position(), robot->global_goal_position()))
                {
                    rest_time = robot->waiting_time() - robot->wait_time_by_picking();
                    return rest_time;
                }
                else
                {
                    rest_time = astar_planning_cost(graph_, robot->graph_id(), robot->target_id()) + robot->waiting_time();
                    return rest_time * (1 + torch::normal(0.25f, 0.5f, 1).clamp(0.0f, 0.5f).item().toFloat());
                }
            }
            else
                return 0.0;
        }
    }
}