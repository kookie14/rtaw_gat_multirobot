#include "allocation/astar_allocation.h"

namespace rds
{
    AStarAllocation::AStarAllocation(SwarmHandle swarm, GraphHandle graph, TaskGeneratorHandle task_gen, RTAWBufferHandle buffer)
    {
        swarm_ = swarm;
        graph_ = graph;
        task_gen_ = task_gen;
        buffer_ = buffer;
    }

    AStarObservation AStarAllocation::observation()
    {
        for (int i = 0; i < swarm_.size(); i++)
        {
            if (swarm_[i]->allocation_state() == FREE)
            {
                TaskHandleList task_queue = task_gen_->task_queue();
                std::vector<float> distances;
                for (int k = 0; k < task_queue.size(); k++)
                {
                    // distances.push_back(astar_planning_cost(graph_, swarm_[i]->graph_id(), task_queue[i]->start_id()));
                    // std::cout << "Neighbor of robot " << swarm_[i]->id() << " ";
                    // for (int neighbor: graph_->neighbors(swarm_[i]->graph_id())) std::cout << neighbor << " ";
                    // std::cout << std::endl;
                    // std::cout << swarm_[i]->graph_id() << " " << task_queue[i]->start_id() << " " << task_queue[i]->target_id() << std::endl;
                    distances.push_back(astar_planning_cost(graph_, swarm_[i]->graph_id(), task_queue[k]->start_id()) + 
                                        astar_planning_cost(graph_, task_queue[k]->start_id(), task_queue[k]->target_id()));
                }

                return AStarObservation(true, i, distances);
            }
            else
                continue;
        }
        return AStarObservation();
    }

    void AStarAllocation::allocation(int robot_id, int task_id)
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

    void AStarAllocation::update_reward()
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