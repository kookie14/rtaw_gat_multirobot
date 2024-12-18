#pragma once
#ifndef SAC_BUFFER_H_
#define SAC_BUFFER_H_

#include "utils/utils.h"
#include <deque>
namespace rds
{
    class SACMiniBuffer
    {
    private:
        int batch_size_;
        int counter_;
        bool has_last_value_;
        torch::DeviceType device_;
        std::vector<bool> reward_states_;
        std::vector<torch::Tensor> node_features_, edge_features_, edge_indices_, start_ids_, target_ids_, robot_ids_, selected_ids_;
        torch::Tensor actions_, rewards_;

    public:
        SACMiniBuffer(int batch_size, torch::DeviceType device);
        void update(torch::Tensor node_feats, torch::Tensor edge_feats, torch::Tensor edge_index, torch::Tensor start_ids,
                    torch::Tensor target_ids, torch::Tensor robot_ids, torch::Tensor selected_ids, torch::Tensor action);
        bool is_full() { return (counter_ == batch_size_); }
        bool is_complete() { return (false == number_exists<bool>(reward_states_, false)); }
        void update_reward(torch::Tensor reward, int reward_idx)
        {
            rewards_[reward_idx] = reward;
            reward_states_[reward_idx] = true;
        }
        void clear();
        int counter() { return counter_; }
        std::vector<bool> reward_states() { return reward_states_; }
        std::vector<torch::Tensor> node_features() { return node_features_; }
        std::vector<torch::Tensor> edge_features() { return edge_features_; }
        std::vector<torch::Tensor> edge_indices() { return edge_indices_; }
        std::vector<torch::Tensor> start_ids() { return start_ids_; }
        std::vector<torch::Tensor> target_ids() { return target_ids_; }
        std::vector<torch::Tensor> robot_ids() { return robot_ids_; }
        std::vector<torch::Tensor> selected_ids() { return selected_ids_; }
        torch::Tensor actions() { return actions_; }
        torch::Tensor rewards() { return rewards_; }
    };
    typedef std::shared_ptr<SACMiniBuffer> SACMiniBufferHandle;

    struct BufferOut
    {
        torch::Tensor node_features, edge_features, edge_indices, start_ids, target_ids, robot_ids, selected_ids;
        torch::Tensor next_node_features, next_edge_features, next_edge_indices, next_start_ids, next_target_ids;
        torch::Tensor next_robot_ids, next_selected_ids;
        torch::Tensor actions, rewards;
        BufferOut()
        {

        }
        BufferOut(torch::Tensor node_features_, torch::Tensor edge_features_,
                  torch::Tensor edge_indices_, torch::Tensor start_ids_,
                  torch::Tensor target_ids_, torch::Tensor robot_ids_,
                  torch::Tensor selected_ids_, torch::Tensor next_node_features_,
                  torch::Tensor next_edge_features_, torch::Tensor next_edge_indices_,
                  torch::Tensor next_start_ids_, torch::Tensor next_target_ids_,
                  torch::Tensor next_robot_ids_, torch::Tensor next_selected_ids_,
                  torch::Tensor actions_, torch::Tensor rewards_)
        {
            node_features = node_features_;
            edge_features = edge_features_;
            edge_indices = edge_indices_;
            start_ids = start_ids_;
            target_ids = target_ids_;
            robot_ids = robot_ids_;
            selected_ids = selected_ids_;

            next_node_features = next_node_features_;
            next_edge_features = next_edge_features_;
            next_edge_indices = next_edge_indices_;
            next_start_ids = next_start_ids_;
            next_target_ids = next_target_ids_;
            next_robot_ids = next_robot_ids_;
            next_selected_ids = next_selected_ids_;

            actions = actions_;
            rewards = rewards_;
        }
    };

    class SACBuffer
    {
    public:
        std::vector<torch::Tensor> node_features, edge_features, edge_indices, start_ids, target_ids, robot_ids, selected_ids;
        std::vector<torch::Tensor> next_node_features, next_edge_features, next_edge_indices, next_start_ids, next_target_ids;
        std::vector<torch::Tensor> next_robot_ids, next_selected_ids;
        std::vector<torch::Tensor> actions, rewards;
        int buffer_size;
        int batch_size;
        torch::DeviceType device;

    public:
        SACBuffer(int buffer_size_, int batch_size_, torch::DeviceType device_);
        void add_experience(SACMiniBufferHandle mini_buffer);
        void add_experience(torch::Tensor node_features_, torch::Tensor edge_features_,
                            torch::Tensor edge_indices_, torch::Tensor start_ids_,
                            torch::Tensor target_ids_, torch::Tensor robot_ids_,
                            torch::Tensor selected_ids_, torch::Tensor next_node_features_,
                            torch::Tensor next_edge_features_, torch::Tensor next_edge_indices_,
                            torch::Tensor next_start_ids_, torch::Tensor next_target_ids_,
                            torch::Tensor next_robot_ids_, torch::Tensor next_selected_ids_,
                            torch::Tensor actions_, torch::Tensor rewards_);
        BufferOut sample();
    };
}
#endif