#pragma once
#ifndef TASK_GENERATOR_H_
#define TASK_GENERATOR_H_

#include "utils/type_define.h"

namespace rds
{
    class TaskGenerator
    {
    private:
        int num_task_queue_; // number of tasks in queue
        int num_priority_; // number of task priorities (0, 1, etc)
        int num_type_; // number of task types 
        float min_mass_, max_mass_; // minimum mass and maximum mass
        GraphHandle graph_; // factory graph
        TaskHandleList task_queue_; // task queue contain tasks in factory
        Vertices w_vertices_, s_vertices_; // working and storage vertices
        std::unordered_map<int, int> working_start_ids_; // start ids are assigned to robots
        std::unordered_map<int, int> working_target_ids_; // target ids are assigned to robots
        std::vector<int> w_ids_; // ID of working vertex
        std::vector<int> s_ids_; // ID of storage vertex
        std::vector<int> task_ids_; // ID of task vertex (working + storage)
        int task_counter_; 
        int max_counter_;
        int mode_;
        std::string data_path_;
        std::vector<std::vector<int>> task_ids_deterministic_;
        int task_deterministic_counter_;
    public:
        TaskGenerator(int num_task_queue, int num_priority, int num_type, float min_mass, 
                    float max_mass, GraphHandle graph, int mode, std::string data_path="");
        void reset();
        TaskHandleList task_queue(){return task_queue_;}
        TaskHandle task(int task_id, int robot_id, bool update_task_queue = true);
        void add_task();
        std::vector<int> filter_vector(std::vector<int> v, int type);
        std::pair<std::vector<int>, std::vector<int>> filter_vector(std::vector<int> working_v, std::vector<int> storage_v);
        void add_task(int type, int priority, float mass, int start_id, int target_id);
        void extract_working_and_storage_vertices();
        void generate_task_queue();
        float max_mass(){return max_mass_;}
        int max_priority(){return num_priority_;}
    };
    typedef std::shared_ptr<TaskGenerator> TaskGeneratorHandle;
}

#endif