#pragma once
#ifndef GNN_ALLOCATION_H_
#define GNN_ALLOCATION_H_

#include "robot/robot.h"
#include "allocation/task_generator.h"
#include "utils/type_define.h"
#include "policy/buffers.h"

namespace rds
{
    struct GNNAllocationObservation
    {
        bool state;
        int robot_id;
        torch::Tensor node_features, edge_features, edge_index, start_ids, target_ids, robot_ids, selected_ids;
        GNNAllocationObservation(bool state_ = false, int robot_id_ = -1, torch::Tensor node_features_ = torch::zeros(1),
                        torch::Tensor edge_features_ = torch::zeros(1), torch::Tensor edge_index_ = torch::zeros(1),
                        torch::Tensor start_ids_ = torch::zeros(1), torch::Tensor target_ids_ = torch::zeros(1),
                        torch::Tensor robot_ids_ = torch::zeros(1), torch::Tensor selected_ids_ = torch::zeros(1))
        {
            state = state_;
            robot_id = robot_id_;
            node_features = node_features_;
            edge_features = edge_features_;
            edge_index = edge_index_;
            start_ids = start_ids_;
            target_ids = target_ids_;
            robot_ids = robot_ids_;
            selected_ids = selected_ids_;
        }
    };
    class GNNAllocation
    {
    private:
        SwarmHandle swarm_;
        GraphHandle robot_graph_;
        GraphHandle graph_;
        TaskGeneratorHandle task_gen_;
        float map_length_, map_width_;
        float dt_;
        float max_path_cost_;
        int mode_;
        torch::DeviceType device_;
        torch::Tensor fixed_node_features_, fixed_edge_features_, fixed_edge_index_;
    public:
        GNNAllocationBufferHandle buffer_;
        GNNAllocation(SwarmHandle swarm, GraphHandle graph, TaskGeneratorHandle task_gen, GNNAllocationBufferHandle buffer,
                       float map_length, float map_width, torch::DeviceType device, float dt = 0.1, int mode = TRAINING);
        void calculate_fixed_node_features();
        void calculate_fixed_edge_features();
        GNNAllocationObservation observation();
        void allocation(int robot_id, int task_id);
        void update_reward();
        std::vector<torch::Tensor> calculate_observation(RobotHandle selected);
        std::vector<torch::Tensor> calculate_observation_ver2(RobotHandle selected);
    };
    typedef std::shared_ptr<GNNAllocation> GNNAllocationHandle;
}

#endif