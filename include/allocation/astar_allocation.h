#pragma once
#ifndef RTAW_ALLOCATION_H_
#define RTAW_ALLOCATION_H_

#include "robot/robot.h"
#include "allocation/task_generator.h"
#include "utils/type_define.h"
#include "policy/buffers.h"

namespace rds
{
    struct AStarObservation
    {
        bool state;
        int robot_id;
        std::vector<float> distances;
        AStarObservation(bool state_ = false, int robot_id_ = -1, std::vector<float> distances_ = {})
        {
            state = state_;
            robot_id = robot_id_;
            distances = distances_;
        }
    };
    class AStarAllocation
    {
    private:
        SwarmHandle swarm_;
        GraphHandle graph_;
        TaskGeneratorHandle task_gen_;
        float map_length_, map_width_;
        float dt_;
        torch::DeviceType device_;

    public:
        RTAWBufferHandle buffer_;
        AStarAllocation(SwarmHandle swarm, GraphHandle graph, TaskGeneratorHandle task_gen, RTAWBufferHandle buffer);
        AStarObservation observation();
        void allocation(int robot_id, int task_id);
        void update_reward();
    };
    typedef std::shared_ptr<AStarAllocation> AStarAllocationHandle;
}

#endif