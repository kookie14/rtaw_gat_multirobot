#include "utils/map_and_graph.h"
namespace rds
{
    std::pair<std::vector<int>, std::vector<Point2D>> astar_planning(std::shared_ptr<Graph> graph, int start_id, int target_id)
    {
        PriorityQueue prior_queue;
        prior_queue.push(AStarItem(start_id, {start_id}, 0.0f), 0);
        std::unordered_set<int> visited;

        while (!prior_queue.isEmpty())
        {
            AStarItem item = prior_queue.pop();
            int current_state = item.current_state();
            std::vector<int> actions = item.actions();
            float cost = item.cost();
            if (visited.find(current_state) == visited.end())
            {
                visited.insert(current_state);

                if (current_state == target_id)
                {
                    std::pair<std::vector<int>, std::vector<Point2D>> route;
                    route.first.push_back(start_id);
                    route.second.push_back(graph->vertex(start_id).position());

                    for (size_t i = 1; i < actions.size() - 1; ++i)
                    {
                        // float angle1 = angle_by_two_point(graph->vertex(actions[i - 1]).position(), graph->vertex(actions[i]).position());
                        // float angle2 = angle_by_two_point(graph->vertex(actions[i]).position(), graph->vertex(actions[i + 1]).position());

                        // if (angle1 != angle2)
                        // {
                        route.first.push_back(graph->vertex(actions[i]).id());
                        route.second.push_back(graph->vertex(actions[i]).position());
                        // }
                    }
                    route.first.push_back(target_id);
                    route.second.push_back(graph->vertex(target_id).position());
                    if (route.first.size() > 2)
                    {
                        if (number_exists<int>(graph->neighbors(start_id), route.first[2]))
                        {
                            route.first.erase(route.first.begin() + 1);
                            route.second.erase(route.second.begin() + 1);
                        }
                    }
                    return route;
                }
                else
                {
                    std::vector<std::pair<int, float>> children = graph->astar_neighbors(current_state);

                    for (const std::pair<int, float> &child : children)
                    {
                        float heuristic_value = manhattan_distance(graph->vertex(child.first).position(), graph->vertex(target_id).position());
                        std::vector<int> actions_ = actions;
                        actions_.push_back(child.first);
                        prior_queue.update(AStarItem(child.first, actions_, cost + child.second), child.second + cost + heuristic_value);
                    }
                }
            }
        }
        return std::pair<std::vector<int>, std::vector<Point2D>>();
    }

    float astar_planning_cost(std::shared_ptr<Graph> graph, int start_id, int target_id)
    {
        PriorityQueue prior_queue;
        prior_queue.push(AStarItem(start_id, {start_id}, 0.0f), 0);
        std::unordered_set<int> visited;

        while (!prior_queue.isEmpty())
        {
            AStarItem item = prior_queue.pop();
            int current_state = item.current_state();
            std::vector<int> actions = item.actions();
            float cost = item.cost();
            if (visited.find(current_state) == visited.end())
            {
                visited.insert(current_state);

                if (current_state == target_id)
                {
                    float cost = 0.0f;
                    for (size_t i = 0; i < actions.size() - 1; ++i)
                    {
                        cost += euclidean_distance(graph->vertex(actions[i]).position(), graph->vertex(actions[i + 1]).position());
                    }

                    return cost;
                }
                else
                {
                    std::vector<std::pair<int, float>> children = graph->astar_neighbors(current_state);
                    for (const std::pair<int, float> &child : children)
                    {
                        float heuristic_value = manhattan_distance(graph->vertex(child.first).position(), graph->vertex(target_id).position());
                        std::vector<int> actions_ = actions;
                        actions_.push_back(child.first);
                        prior_queue.update(AStarItem(child.first, actions_, cost + child.second), child.second + cost + heuristic_value);
                    }
                }
            }
        }
        return 0.0f;
    }

    float average_route_velocity(std::shared_ptr<Graph> graph, std::vector<int> route_ids)
    {
        float sum_velocity = 0.0f;
        float sum_distance = 0.0f;
        for (int i = 0; i < route_ids.size() - 1; i++)
        {
            sum_velocity += graph->edge(route_ids[i], route_ids[i + 1]).velocity() * graph->edge(route_ids[i], route_ids[i + 1]).distance();
            sum_distance += graph->edge(route_ids[i], route_ids[i + 1]).distance();
        }
        return (float)sum_velocity / sum_distance;
    }

    std::vector<std::vector<float>> calculate_average_velocity_and_distance_forward(std::shared_ptr<Graph> graph,
                                                                                    std::vector<int> route_ids)
    {
        std::vector<float> average_velocities, distances, directions;
        float sum_velocity = 0.0f;
        float sum_distance = 0.0f;
        for (int i = 0; i < route_ids.size() - 1; i++)
        {
            sum_velocity += graph->edge(route_ids[i], route_ids[i + 1]).velocity() *
                            graph->edge(route_ids[i], route_ids[i + 1]).distance();
            sum_distance += graph->edge(route_ids[i], route_ids[i + 1]).distance();
            average_velocities.push_back(float(sum_velocity / sum_distance));
            distances.push_back(sum_distance);
            directions.push_back(graph->edge(route_ids[i], route_ids[i + 1]).direction());
        }

        return {average_velocities, distances, directions};
    }

    std::vector<std::vector<float>> calculate_average_velocity_and_distance_backward(std::shared_ptr<Graph> graph,
                                                                                     std::vector<int> route_ids)
    {
        std::vector<float> average_velocities(route_ids.size() - 1), distances(route_ids.size() - 1), directions(route_ids.size() - 1);
        float sum_velocity = 0.0f;
        float sum_distance = 0.0f;
        for (int i = route_ids.size() - 1; i >= 1; i--)
        {
            sum_velocity += graph->edge(route_ids[i - 1], route_ids[i]).velocity() *
                            graph->edge(route_ids[i - 1], route_ids[i]).distance();
            sum_distance += graph->edge(route_ids[i - 1], route_ids[i]).distance();
            average_velocities[i-1] = float(sum_velocity / sum_distance);
            distances[i-1] = sum_distance;
            directions[i-1] = graph->edge(route_ids[i-1], route_ids[i]).direction();
        }

        return {average_velocities, distances, directions};
    }

    bool compare_vertex_by_x(const Vertex &v1, const Vertex &v2)
    {
        return v1.x() < v2.x();
    }
    bool compare_vertex_by_y(const Vertex &v1, const Vertex &v2)
    {
        return v1.y() < v2.y();
    }
}