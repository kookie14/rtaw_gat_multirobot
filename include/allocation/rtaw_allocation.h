#pragma once
#ifndef RTAW_ALLOCATION_H_
#define RTAW_ALLOCATION_H_

#include "robot/robot.h"
#include "allocation/task_generator.h"
#include "utils/type_define.h"
#include "policy/buffers.h"

namespace rds
{
    struct RTAWObservation
    {
        bool state;
        int robot_id;
        torch::Tensor select, robot, task;
        RTAWObservation(bool state_ = false, int robot_id_ = -1, torch::Tensor select_ = torch::zeros(1),
                        torch::Tensor robot_ = torch::zeros(1), torch::Tensor task_ = torch::zeros(1))
        {
            state = state_;
            robot_id = robot_id_;
            select = select_;
            robot = robot_;
            task = task_;
        }
    };
    class RTAWAllocation
    {
    private:
        SwarmHandle swarm_;
        GraphHandle graph_;
        TaskGeneratorHandle task_gen_;
        float map_length_, map_width_;
        float dt_;
        float max_rest_time_, max_path_cost_;
        int mode_;
        torch::DeviceType device_;
    public:
        RTAWBufferHandle buffer_;
        RTAWAllocation(SwarmHandle swarm, GraphHandle graph, TaskGeneratorHandle task_gen, RTAWBufferHandle buffer,
                       float map_length, float map_width, torch::DeviceType device, float dt = 0.1, int mode = TRAINING);
        RTAWObservation observation();
        void allocation(int robot_id, int task_id);
        void update_reward();
        std::vector<torch::Tensor> calculate_observation(RobotHandle selected);
        torch::Tensor calculate_task_observation(RobotHandle selected);
        torch::Tensor calculate_robot_observation();
        float estimate_rest_time(RobotHandle robot);
    };
    typedef std::shared_ptr<RTAWAllocation> RTAWAllocationHandle;
}

#endif