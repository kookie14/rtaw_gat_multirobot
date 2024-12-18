#pragma once
#ifndef DOUBLE_SINGLE_ENVIRONMENT_H_
#define DOUBLE_SINGLE_ENVIRONMENT_H_

#include "env/environment_interface.h"
namespace rds
{
    class DoubleSingleEnvironment : public EnvironmentInterface
    {
    private:
    public:
        DoubleSingleEnvironment(std::string data_path, int num_robots, float robot_max_speed, float robot_max_payload,
                                matplot::axes_handle visual, bool show_edge = false);
        virtual void reset(){};
        virtual SwarmHandle swarm();
        virtual GraphHandle graph_handle();
        virtual Graph graph();
        virtual void controller(float wait_time);
        virtual void visualize();
        virtual Point2D map_center();
        virtual float map_length();
        virtual float map_width();
        virtual float single_line_width();
        virtual float double_line_width();
        virtual float dt();
        virtual void add_all_robot_graph(){};
    };
}
#endif