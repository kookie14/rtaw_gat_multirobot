#pragma once
#ifndef ASTAR_PLANNING_NO_WAITING_H_
#define ASTAR_PLANNING_NO_WAITING_H_

#include "planner/planner_interface.h"

namespace rds
{
    class AStarPlanner : public PlannerInterface
    {
    private:
        SwarmHandle swarm_;
        GraphHandle graph_;
        Point2DHandle map_center_;
        float map_length_, map_width_, single_line_width_, double_line_width_;

    public:
        AStarPlanner(SwarmHandle swarm, GraphHandle graph, Point2D map_center, float map_length,
                    float map_width, float single_line_width, float double_line_width);
        virtual void planning();
    };
}

#endif