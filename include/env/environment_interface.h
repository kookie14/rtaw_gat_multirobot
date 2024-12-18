#pragma once
#ifndef ENVIRONMENT_H_
#define ENVIRONMENT_H_

#include "robot/robot.h"
#include "utils/utils.h"
#include "env/reconstruct_interface.h"
#include "env/double_single_reconstruct.h"
#include "env/single_reconstruct.h"
#include "env/real_reconstruct.h"

namespace rds
{
    class EnvironmentInterface
    {
    public:
        virtual void reset() = 0;
        virtual SwarmHandle swarm() = 0;
        virtual GraphHandle graph_handle() = 0;
        virtual Graph graph() = 0;
        virtual void controller(float wait_time) = 0;
        virtual void visualize() = 0;
        virtual Point2D map_center() = 0;
        virtual float map_length() = 0;
        virtual float map_width() = 0;
        virtual float single_line_width() = 0;
        virtual float double_line_width() = 0;
        virtual float dt() = 0;
        virtual void add_all_robot_graph() = 0;
    };

    typedef std::shared_ptr<EnvironmentInterface> EnvironmentInterfaceHandle;
}
#endif