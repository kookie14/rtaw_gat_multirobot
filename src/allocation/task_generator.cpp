#include "allocation/task_generator.h"

namespace rds
{
    TaskGenerator::TaskGenerator(int num_task_queue, int num_priority, int num_type, float min_mass,
                                 float max_mass, GraphHandle graph, int mode, std::string data_path)
    {
        num_task_queue_ = num_task_queue;
        num_priority_ = num_priority;
        num_type_ = num_type;
        min_mass_ = min_mass;
        max_mass_ = max_mass;
        graph_ = std::make_shared<Graph>(*graph);
        task_counter_ = 1;
        max_counter_ = 20;
        extract_working_and_storage_vertices();
        for (int i = 0; i < w_vertices_.size(); i++)
        {
            w_ids_.push_back(w_vertices_[i].id());
            task_ids_.push_back(w_vertices_[i].id());
        }
        for (int i = 0; i < s_vertices_.size(); i++)
        {
            s_ids_.push_back(s_vertices_[i].id());
            task_ids_.push_back(s_vertices_[i].id());
        }
        mode_ = mode;
        data_path_ = data_path;
        if (mode_ == DETERMINISTIC_MODE)
        {
            std::vector<std::vector<float>> task_ids = load_txt(data_path_ + "/task_ids.txt");
            for (std::vector<float> task : task_ids)
            {
                task_ids_deterministic_.push_back({int(task[0]), int(task[1])});
            }
            task_deterministic_counter_ = 0;
        }
        generate_task_queue();
    }

    void TaskGenerator::reset()
    {
        if (mode_ == DETERMINISTIC_MODE)
        {
            task_deterministic_counter_ = 0;
        }
        task_queue_.clear();
        generate_task_queue();
    }

    TaskHandle TaskGenerator::task(int task_id, int robot_id, bool update_task_queue)
    {
        TaskHandle task = std::make_shared<Task>(*task_queue_[task_id]);
        working_start_ids_[robot_id] = task->start_id();
        working_target_ids_[robot_id] = task->target_id();
        task_queue_.erase(task_queue_.begin() + task_id);
        if (update_task_queue)
            add_task();
        return task;
    }

    void TaskGenerator::add_task()
    {
        if (mode_ == DETERMINISTIC_MODE)
        {
            int type = torch::randint(1, num_type_, 1).item().toInt();
            int priority = torch::randint(1, num_priority_, 1).item().toInt();
            float mass = torch::randint((int)min_mass_, (int)max_mass_, 1).item().toFloat();
            if (task_deterministic_counter_ >= task_ids_deterministic_.size())
            {
                task_deterministic_counter_ = 0;
            }
            task_queue_.push_back(std::make_shared<Task>(type, priority, mass, 
                                                        graph_->vertex(task_ids_deterministic_[task_deterministic_counter_][0]), 
                                                        graph_->vertex(task_ids_deterministic_[task_deterministic_counter_][1])));
            std::pair<std::vector<int>, std::vector<Point2D>> route = astar_planning(graph_, task_queue_.back()->start_id(), task_queue_.back()->target_id());
            task_queue_.back()->set_route(Route(TO_TARGET, route.first, route.second));
            task_deterministic_counter_ += 1;
        }
        else
        {
            int type, priority, start_id, target_id, start_idx;
            float mass;
            if (task_counter_ <= max_counter_ - 2) // working to working
            {
                std::vector<int> working_ids_filtered = filter_vector(w_ids_, WORKING_VERTEX);
                if (working_ids_filtered.size() == 0)
                    return;
                for (int i = 0; i < 100; i++)
                {
                    std::vector<int> working_ids = working_ids_filtered;
                    type = torch::randint(1, num_type_, 1).item().toInt();
                    priority = torch::randint(1, num_priority_, 1).item().toInt();
                    mass = torch::randint((int)min_mass_, (int)max_mass_, 1).item().toFloat();
                    start_idx = torch::randint(0, working_ids.size() - 1, 1).item().toInt();
                    start_id = working_ids[start_idx];
                    working_ids.erase(working_ids.begin() + start_idx);
                    target_id = working_ids[torch::randint(0, working_ids.size() - 1, 1).item().toInt()];
                    if (euclidean_distance(graph_->vertex(start_id).position(), graph_->vertex(target_id).position()) > 5)
                    {
                        task_queue_.push_back(std::make_shared<Task>(type, priority, mass, graph_->vertex(start_id), graph_->vertex(target_id)));
                        std::pair<std::vector<int>, std::vector<Point2D>> route = astar_planning(graph_, start_id, target_id);
                        task_queue_.back()->set_route(Route(TO_TARGET, route.first, route.second));
                        break;
                    }
                }
                task_counter_ += 1;
            }
            else if (task_counter_ > max_counter_ - 2 && task_counter_ <= max_counter_ - 1) // working to storage
            {
                std::pair<std::vector<int>, std::vector<int>> filter_vec = filter_vector(w_ids_, s_ids_);
                if (filter_vec.first.size() == 0 or filter_vec.second.size() == 0)
                    return;
                for (int i = 0; i < 100; i++)
                {
                    type = torch::randint(1, num_type_, 1).item().toInt();
                    priority = torch::randint(1, num_priority_, 1).item().toInt();
                    mass = torch::randint((int)min_mass_, (int)max_mass_, 1).item().toFloat();
                    start_id = filter_vec.first[torch::randint(0, filter_vec.first.size() - 1, 1).item().toInt()];
                    target_id = filter_vec.second[torch::randint(0, filter_vec.second.size() - 1, 1).item().toInt()];
                    if (euclidean_distance(graph_->vertex(start_id).position(), graph_->vertex(target_id).position()) > 5)
                    {
                        task_queue_.push_back(std::make_shared<Task>(type, priority, mass, graph_->vertex(start_id), graph_->vertex(target_id)));
                        std::pair<std::vector<int>, std::vector<Point2D>> route = astar_planning(graph_, start_id, target_id);
                        task_queue_.back()->set_route(Route(TO_TARGET, route.first, route.second));
                        break;
                    }
                }
                task_counter_ += 1;
            }
            else if (task_counter_ > max_counter_ - 1 && task_counter_ <= max_counter_)
            {
                std::pair<std::vector<int>, std::vector<int>> filter_vec = filter_vector(w_ids_, s_ids_);
                if (filter_vec.first.size() == 0 or filter_vec.second.size() == 0)
                    return;
                for (int i = 0; i < 100; i++)
                {
                    type = torch::randint(1, num_type_, 1).item().toInt();
                    priority = torch::randint(1, num_priority_, 1).item().toInt();
                    mass = torch::randint((int)min_mass_, (int)max_mass_, 1).item().toFloat();
                    start_id = filter_vec.second[torch::randint(0, filter_vec.second.size() - 1, 1).item().toInt()];
                    target_id = filter_vec.first[torch::randint(0, filter_vec.first.size() - 1, 1).item().toInt()];
                    if (euclidean_distance(graph_->vertex(start_id).position(), graph_->vertex(target_id).position()) > 5)
                    {
                        task_queue_.push_back(std::make_shared<Task>(type, priority, mass, graph_->vertex(start_id), graph_->vertex(target_id)));
                        std::pair<std::vector<int>, std::vector<Point2D>> route = astar_planning(graph_, start_id, target_id);
                        task_queue_.back()->set_route(Route(TO_TARGET, route.first, route.second));
                        break;
                    }
                }
                task_counter_ = 1;
            }
        }
    }
    void TaskGenerator::add_task(int type, int priority, float mass, int start_id, int target_id)
    {
        if (number_exists(task_ids_, start_id) && number_exists(task_ids_, target_id))
        {
            task_queue_.push_back(std::make_shared<Task>(type, priority, mass, graph_->vertex(start_id), graph_->vertex(target_id)));
        }
    }

    std::vector<int> TaskGenerator::filter_vector(std::vector<int> v, int type)
    {
        for (auto task : working_start_ids_)
        {
            if (graph_->vertex(task.second).type() == type)
            {
                auto it = std::remove(v.begin(), v.end(), task.second);
                v.erase(it, v.end());
            }
        }
        for (auto task : working_target_ids_)
        {
            if (graph_->vertex(task.second).type() == type)
            {
                auto it = std::remove(v.begin(), v.end(), task.second);
                v.erase(it, v.end());
            }
        }
        for (TaskHandle task : task_queue_)
        {
            if (graph_->vertex(task->start_id()).type() == type)
            {
                auto it = std::remove(v.begin(), v.end(), task->start_id());
                v.erase(it, v.end());
            }
            if (graph_->vertex(task->target_id()).type() == type)
            {
                auto it = std::remove(v.begin(), v.end(), task->target_id());
                v.erase(it, v.end());
            }
        }

        return v;
    }

    std::pair<std::vector<int>, std::vector<int>> TaskGenerator::filter_vector(std::vector<int> working_v, std::vector<int> storage_v)
    {
        for (auto task : working_start_ids_)
        {
            if (graph_->vertex(task.second).type() == WORKING_VERTEX)
            {
                auto it = std::remove(working_v.begin(), working_v.end(), task.second);
                working_v.erase(it, working_v.end());
            }
            if (graph_->vertex(task.second).type() == STORAGE_VERTEX)
            {
                auto it = std::remove(storage_v.begin(), storage_v.end(), task.second);
                storage_v.erase(it, storage_v.end());
            }
        }
        for (auto task : working_target_ids_)
        {
            if (graph_->vertex(task.second).type() == WORKING_VERTEX)
            {
                auto it = std::remove(working_v.begin(), working_v.end(), task.second);
                working_v.erase(it, working_v.end());
            }
            if (graph_->vertex(task.second).type() == STORAGE_VERTEX)
            {
                auto it = std::remove(storage_v.begin(), storage_v.end(), task.second);
                storage_v.erase(it, storage_v.end());
            }
        }
        for (TaskHandle task : task_queue_)
        {
            if (graph_->vertex(task->start_id()).type() == WORKING_VERTEX)
            {
                auto it = std::remove(working_v.begin(), working_v.end(), task->start_id());
                working_v.erase(it, working_v.end());
            }
            if (graph_->vertex(task->target_id()).type() == WORKING_VERTEX)
            {
                auto it = std::remove(working_v.begin(), working_v.end(), task->target_id());
                working_v.erase(it, working_v.end());
            }
            if (graph_->vertex(task->start_id()).type() == STORAGE_VERTEX)
            {
                auto it = std::remove(storage_v.begin(), storage_v.end(), task->start_id());
                storage_v.erase(it, storage_v.end());
            }
            if (graph_->vertex(task->target_id()).type() == STORAGE_VERTEX)
            {
                auto it = std::remove(storage_v.begin(), storage_v.end(), task->target_id());
                storage_v.erase(it, storage_v.end());
            }
        }

        std::pair<std::vector<int>, std::vector<int>> out;
        out.first = working_v;
        out.second = storage_v;

        return out;
    }

    void TaskGenerator::extract_working_and_storage_vertices()
    {
        Vertices graph_vertices = graph_->vertices();
        for (int i = 0; i < graph_vertices.size(); i++)
        {
            if (graph_vertices[i].type() == WORKING_VERTEX)
                w_vertices_.push_back(graph_->vertex(i));
            if (graph_vertices[i].type() == STORAGE_VERTEX)
                s_vertices_.push_back(graph_->vertex(i));
        }
    }

    void TaskGenerator::generate_task_queue()
    {
        for (int i = 0; i < num_task_queue_; i++)
        {
            add_task();
        }
    }
}