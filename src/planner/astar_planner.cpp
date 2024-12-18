#include "planner/astar_planner.h"

namespace rds
{
    AStarPlanner::AStarPlanner(SwarmHandle swarm, GraphHandle graph, Point2D map_center, float map_length,
                               float map_width, float single_line_width, float double_line_width)
    {
        swarm_ = swarm;
        graph_ = graph;
        map_center_ = std::make_shared<Point2D>(map_center);
        map_length_ = map_length;
        map_width_ = map_width;
        single_line_width_ = single_line_width;
        double_line_width_ = double_line_width;
    }

    void AStarPlanner::planning()
    {
        // Initialize path for each robot
        std::pair<std::vector<int>, std::vector<Point2D>> route;
        for (int i = 0; i < swarm_.size(); i++)
        {
            if (swarm_[i]->has_task() && !swarm_[i]->has_route())
            {
                route = astar_planning(graph_, swarm_[i]->graph_id(), swarm_[i]->global_goal_id());
                swarm_[i]->set_route(swarm_[i]->global_goal_type(), route.first, route.second);
            }
        }
        // Solve conflict path between robots
    }
}