#pragma once
#ifndef BUFFERS_H_
#define BUFFERS_H_

#include <torch/torch.h>
#include <vector>
#include "utils/utils.h"
namespace rds
{
    class RTAWBuffer
    {
    private:
        int batch_size_;
        int num_robots_;
        int num_tasks_;
        float gamma_;
        float lambda_;
        int counter_;
        bool has_last_value_;
        torch::DeviceType device_;
        std::vector<bool> reward_states_;
        torch::Tensor selects_, robots_, tasks_, actions_, rewards_, values_, log_probs_, advantages_, returns_,last_value_;
    public:
        RTAWBuffer(int batch_size, int num_robots, int num_tasks, float gamma, float lambda, torch::DeviceType device);
        void update(torch::Tensor select, torch::Tensor robot, torch::Tensor task, torch::Tensor action, torch::Tensor value, torch::Tensor log_prob);
        bool is_full() { return (counter_ == batch_size_); }
        bool is_complete() { return (false == number_exists<bool>(reward_states_, false)); }
        void update_reward(torch::Tensor reward, int reward_idx)
        {
            rewards_[reward_idx] = reward;
            reward_states_[reward_idx] = true;
        }
        void set_last_value(torch::Tensor last_value) { last_value_ = last_value;}
        bool has_last_value() { return has_last_value_; }
        void set_has_last_value(bool has_last_value) { has_last_value_ = has_last_value;}
        void estimate_advantage();
        void clear();
        int counter() { return counter_; }
        std::vector<bool> reward_states() { return reward_states_; }
        torch::Tensor selects() {return selects_; }
        torch::Tensor robots() {return robots_; }
        torch::Tensor tasks() {return tasks_; }
        torch::Tensor actions() {return actions_; }
        torch::Tensor rewards() {return rewards_; }
        torch::Tensor values() {return values_; }
        torch::Tensor log_probs() {return log_probs_; }
        torch::Tensor advantages() {return advantages_; }
        torch::Tensor returns() {return returns_; }
    };
    typedef std::shared_ptr<RTAWBuffer> RTAWBufferHandle;

    struct RTAWBuffers
    {
        torch::Tensor selects, robots, tasks, actions, rewards, values, log_probs, advantages, returns;
        RTAWBuffers(std::vector<RTAWBufferHandle> buffers)
        {
            selects = buffers.front()->selects();
            robots = buffers.front()->robots();
            tasks = buffers.front()->tasks();
            actions = buffers.front()->actions();
            rewards = buffers.front()->rewards();
            values = buffers.front()->values();
            log_probs = buffers.front()->log_probs();
            advantages = buffers.front()->advantages();
            returns = buffers.front()->returns();
            for (int i = 1; i < buffers.size(); i++)
            {
                selects = torch::cat({selects, buffers[i]->selects()}, 0);
                robots = torch::cat({robots, buffers[i]->robots()}, 0);
                tasks = torch::cat({tasks, buffers[i]->tasks()}, 0);
                actions = torch::cat({actions, buffers[i]->actions()}, 0);
                rewards = torch::cat({rewards, buffers[i]->rewards()}, 0);
                values = torch::cat({values, buffers[i]->values()}, 0);
                log_probs = torch::cat({log_probs, buffers[i]->log_probs()}, 0);
                advantages = torch::cat({advantages, buffers[i]->advantages()}, 0);
                returns = torch::cat({returns, buffers[i]->returns()}, 0);
            }
        }
    };

    class GNNAllocationBuffer
    {
    private:
        int batch_size_;
        float gamma_;
        float lambda_;
        int counter_;
        bool has_last_value_;
        torch::DeviceType device_;
        std::vector<bool> reward_states_;
        std::vector<torch::Tensor> node_features_, edge_features_, edge_indices_, start_ids_, target_ids_, robot_ids_, selected_ids_;
        torch::Tensor actions_, rewards_, values_, log_probs_, advantages_, returns_, last_value_;
    public:
        GNNAllocationBuffer(int batch_size, float gamma, float lambda, torch::DeviceType device);
        void update(torch::Tensor node_feats, torch::Tensor edge_feats, torch::Tensor edge_index, torch::Tensor start_ids, 
                    torch::Tensor target_ids, torch::Tensor robot_ids, torch::Tensor selected_ids, 
                    torch::Tensor action, torch::Tensor value, torch::Tensor log_prob);
        bool is_full() { return (counter_ == batch_size_); }
        bool is_complete() { return (false == number_exists<bool>(reward_states_, false)); }
        void update_reward(torch::Tensor reward, int reward_idx)
        {
            rewards_[reward_idx] = reward;
            reward_states_[reward_idx] = true;
        }
        void set_last_value(torch::Tensor last_value) { last_value_ = last_value;}
        bool has_last_value() { return has_last_value_; }
        void set_has_last_value(bool has_last_value) { has_last_value_ = has_last_value;}
        void estimate_advantage();
        void clear();
        int counter() { return counter_; }
        std::vector<bool> reward_states() { return reward_states_; }
        std::vector<torch::Tensor>* node_features() {return &node_features_; }
        std::vector<torch::Tensor>* edge_features() {return &edge_features_; }
        std::vector<torch::Tensor>* edge_indices() {return &edge_indices_; }
        std::vector<torch::Tensor>* start_ids() {return &start_ids_; }
        std::vector<torch::Tensor>* target_ids() {return &target_ids_; }
        std::vector<torch::Tensor>* robot_ids() {return &robot_ids_; }
        std::vector<torch::Tensor>* selected_ids() {return &selected_ids_; }
        torch::Tensor actions() {return actions_; }
        torch::Tensor rewards() {return rewards_; }
        torch::Tensor values() {return values_; }
        torch::Tensor log_probs() {return log_probs_; }
        torch::Tensor advantages() {return advantages_; }
        torch::Tensor returns() {return returns_; }
    };
    typedef std::shared_ptr<GNNAllocationBuffer> GNNAllocationBufferHandle;

    struct GNNAllocationBuffers
    {
        std::vector<torch::Tensor> node_features, edge_features, edge_indices, start_ids, target_ids, robot_ids, selected_ids;
        torch::Tensor actions, rewards, values, log_probs, advantages, returns;
        GNNAllocationBuffers(std::vector<GNNAllocationBufferHandle> buffers)
        {
            node_features = *buffers.front()->node_features();
            edge_features = *buffers.front()->edge_features();
            edge_indices = *buffers.front()->edge_indices();
            start_ids = *buffers.front()->start_ids();
            target_ids = *buffers.front()->target_ids();
            robot_ids = *buffers.front()->robot_ids();
            selected_ids = *buffers.front()->selected_ids();
            actions = buffers.front()->actions();
            rewards = buffers.front()->rewards();
            values = buffers.front()->values();
            log_probs = buffers.front()->log_probs();
            advantages = buffers.front()->advantages();
            returns = buffers.front()->returns();
            for (int i = 1; i < buffers.size(); i++)
            {
                node_features.insert(node_features.end(), buffers[i]->node_features()->begin(), buffers[i]->node_features()->end());
                edge_features.insert(edge_features.end(), buffers[i]->edge_features()->begin(), buffers[i]->edge_features()->end());
                edge_indices.insert(edge_indices.end(), buffers[i]->edge_indices()->begin(), buffers[i]->edge_indices()->end());
                start_ids.insert(start_ids.end(), buffers[i]->start_ids()->begin(), buffers[i]->start_ids()->end());
                target_ids.insert(target_ids.end(), buffers[i]->target_ids()->begin(), buffers[i]->target_ids()->end());
                robot_ids.insert(robot_ids.end(), buffers[i]->robot_ids()->begin(), buffers[i]->robot_ids()->end());
                selected_ids.insert(selected_ids.end(), buffers[i]->selected_ids()->begin(), buffers[i]->selected_ids()->end());
                actions = torch::cat({actions, buffers[i]->actions()}, 0);
                rewards = torch::cat({rewards, buffers[i]->rewards()}, 0);
                values = torch::cat({values, buffers[i]->values()}, 0);
                log_probs = torch::cat({log_probs, buffers[i]->log_probs()}, 0);
                advantages = torch::cat({advantages, buffers[i]->advantages()}, 0);
                returns = torch::cat({returns, buffers[i]->returns()}, 0);
            }
        }
    };
}

#endif