#include "policy/sac_buffers.h"

namespace rds
{
    SACMiniBuffer::SACMiniBuffer(int batch_size, torch::DeviceType device)
    {
        batch_size_ = batch_size + 1;
        counter_ = 0;
        has_last_value_ = false;
        device_ = device;
        for (int i = 0; i < batch_size_; ++i)
        {
            reward_states_.push_back(false);
        }
        {
            torch::NoGradGuard no_grad;
            node_features_.clear();
            edge_features_.clear();
            edge_indices_.clear();
            start_ids_.clear();
            target_ids_.clear();
            robot_ids_.clear();
            selected_ids_.clear();
            actions_ = torch::zeros(batch_size_).to(device_);
            rewards_ = torch::zeros(batch_size_).to(device_);
        }
    }

    void SACMiniBuffer::update(torch::Tensor node_feats, torch::Tensor edge_feats, torch::Tensor edge_index, torch::Tensor start_ids,
                               torch::Tensor target_ids, torch::Tensor robot_ids, torch::Tensor selected_ids,
                               torch::Tensor action)
    {
        {
            torch::NoGradGuard no_grad;
            node_features_.push_back(node_feats);
            edge_features_.push_back(edge_feats);
            edge_indices_.push_back(edge_index);
            start_ids_.push_back(start_ids.reshape({1, -1}));
            target_ids_.push_back(target_ids.reshape({1, -1}));
            robot_ids_.push_back(robot_ids);
            selected_ids_.push_back(selected_ids);
            actions_[counter_] = action[0];
            counter_ += 1;
        }
    }

    void SACMiniBuffer::clear()
    {
        counter_ = 0;

        for (int i = 0; i < batch_size_; i++)
        {
            reward_states_[i] = false;
        }
        {
            torch::NoGradGuard no_grad;
            node_features_.clear();
            edge_features_.clear();
            edge_indices_.clear();
            start_ids_.clear();
            target_ids_.clear();
            robot_ids_.clear();
            selected_ids_.clear();
            actions_ = torch::zeros(batch_size_).to(device_);
            rewards_ = torch::zeros(batch_size_).to(device_);
        }
    }

    SACBuffer::SACBuffer(int buffer_size_, int batch_size_, torch::DeviceType device_)
    {
        buffer_size = buffer_size_;
        batch_size = batch_size_;
        device = device_;
    }
    void SACBuffer::add_experience(SACMiniBufferHandle mini_buffer)
    {
        std::vector<torch::Tensor> node_features_ = mini_buffer->node_features();
        std::vector<torch::Tensor> edge_features_ = mini_buffer->edge_features();
        std::vector<torch::Tensor> edge_indices_ = mini_buffer->edge_indices();
        std::vector<torch::Tensor> start_ids_ = mini_buffer->start_ids();
        std::vector<torch::Tensor> target_ids_ = mini_buffer->target_ids();
        std::vector<torch::Tensor> robot_ids_ = mini_buffer->robot_ids();
        std::vector<torch::Tensor> selected_ids_ = mini_buffer->selected_ids();
        torch::Tensor actions_ = mini_buffer->actions();
        torch::Tensor rewards_ = mini_buffer->rewards();

        int num_elements = node_features.size() + node_features_.size() - 1 - buffer_size;
        if (num_elements > 0)
        {
        }
        node_features.insert(node_features.end(), node_features_.begin(), node_features_.end() - 1);
        edge_features.insert(edge_features.end(), edge_features_.begin(), edge_features_.end() - 1);
        edge_indices.insert(edge_indices.end(), edge_indices_.begin(), edge_indices_.end() - 1);
        start_ids.insert(start_ids.end(), start_ids_.begin(), start_ids_.end() - 1);
        target_ids.insert(target_ids.end(), target_ids_.begin(), target_ids_.end() - 1);
        robot_ids.insert(robot_ids.end(), robot_ids_.begin(), robot_ids_.end() - 1);
        selected_ids.insert(selected_ids.end(), selected_ids_.begin(), selected_ids_.end() - 1);

        next_node_features.insert(next_node_features.end(), node_features_.begin() + 1, node_features_.end());
        next_edge_features.insert(next_edge_features.end(), edge_features_.begin() + 1, edge_features_.end());
        next_edge_indices.insert(next_edge_indices.end(), edge_indices_.begin() + 1, edge_indices_.end());
        next_start_ids.insert(next_start_ids.end(), start_ids_.begin() + 1, start_ids_.end());
        next_target_ids.insert(next_target_ids.end(), target_ids_.begin() + 1, target_ids_.end());
        next_robot_ids.insert(next_robot_ids.end(), robot_ids_.begin() + 1, robot_ids_.end());
        next_selected_ids.insert(next_selected_ids.end(), selected_ids_.begin() + 1, selected_ids_.end());

        for (int i = 0; i < actions_.size(0) - 1; i++)
        {
            actions.push_back(actions_[i].reshape(1));
            rewards.push_back(rewards_[i].reshape(1));
        }
    }
    void SACBuffer::add_experience(torch::Tensor node_features_, torch::Tensor edge_features_,
                                   torch::Tensor edge_indices_, torch::Tensor start_ids_,
                                   torch::Tensor target_ids_, torch::Tensor robot_ids_,
                                   torch::Tensor selected_ids_, torch::Tensor next_node_features_,
                                   torch::Tensor next_edge_features_, torch::Tensor next_edge_indices_,
                                   torch::Tensor next_start_ids_, torch::Tensor next_target_ids_,
                                   torch::Tensor next_robot_ids_, torch::Tensor next_selected_ids_,
                                   torch::Tensor actions_, torch::Tensor rewards_)
    {
    }
    BufferOut SACBuffer::sample()
    {
        torch::Tensor sample_ids_torch = torch::randint(0, node_features.size() - 1, batch_size).toType(torch::kInt32);
        std::vector<int> sample_ids(sample_ids_torch.data_ptr<int>(), sample_ids_torch.data_ptr<int>() + sample_ids_torch.numel());

        BufferOut sample_buffer;
        sample_buffer.node_features = node_features[sample_ids[0]];
        sample_buffer.edge_features = edge_features[sample_ids[0]];
        sample_buffer.edge_indices = edge_indices[sample_ids[0]];
        sample_buffer.start_ids = start_ids[sample_ids[0]];
        sample_buffer.target_ids = target_ids[sample_ids[0]];
        sample_buffer.robot_ids = robot_ids[sample_ids[0]];
        sample_buffer.selected_ids = selected_ids[sample_ids[0]];

        sample_buffer.next_node_features = next_node_features[sample_ids[0]];
        sample_buffer.next_edge_features = next_edge_features[sample_ids[0]];
        sample_buffer.next_edge_indices = next_edge_indices[sample_ids[0]];
        sample_buffer.next_start_ids = next_start_ids[sample_ids[0]];
        sample_buffer.next_target_ids = next_target_ids[sample_ids[0]];
        sample_buffer.next_robot_ids = next_robot_ids[sample_ids[0]];
        sample_buffer.next_selected_ids = next_selected_ids[sample_ids[0]];

        sample_buffer.actions = actions[sample_ids[0]];
        sample_buffer.rewards = rewards[sample_ids[0]];
        for (int i = 1; i < sample_ids.size(); i++)
        {
            sample_buffer.edge_indices = torch::cat({sample_buffer.edge_indices, edge_indices[sample_ids[i]] + sample_buffer.node_features.size(0)}, 1);
            sample_buffer.node_features = torch::cat({sample_buffer.node_features, node_features[sample_ids[i]]});
            sample_buffer.edge_features = torch::cat({sample_buffer.edge_features, edge_features[sample_ids[i]]});

            sample_buffer.start_ids = torch::cat({sample_buffer.start_ids, start_ids[sample_ids[i]]});
            sample_buffer.target_ids = torch::cat({sample_buffer.target_ids, target_ids[sample_ids[i]]});
            sample_buffer.robot_ids = torch::cat({sample_buffer.robot_ids, robot_ids[sample_ids[i]]});
            sample_buffer.selected_ids = torch::cat({sample_buffer.selected_ids, selected_ids[sample_ids[i]]});
            
            sample_buffer.next_edge_indices = torch::cat({sample_buffer.next_edge_indices, next_edge_indices[sample_ids[i]] + sample_buffer.next_node_features.size(0)}, 1);
            sample_buffer.next_node_features = torch::cat({sample_buffer.next_node_features, next_node_features[sample_ids[i]]});
            sample_buffer.next_edge_features = torch::cat({sample_buffer.next_edge_features, next_edge_features[sample_ids[i]]});
            
            sample_buffer.next_start_ids = torch::cat({sample_buffer.next_start_ids, next_start_ids[sample_ids[i]]});
            sample_buffer.next_target_ids = torch::cat({sample_buffer.next_target_ids, next_target_ids[sample_ids[i]]});
            sample_buffer.next_robot_ids = torch::cat({sample_buffer.next_robot_ids, next_robot_ids[sample_ids[i]]});
            sample_buffer.next_selected_ids = torch::cat({sample_buffer.next_selected_ids, next_selected_ids[sample_ids[i]]});
            
            sample_buffer.actions = torch::cat({sample_buffer.actions, actions[sample_ids[i]]});
            sample_buffer.rewards = torch::cat({sample_buffer.rewards, rewards[sample_ids[i]]});
        }

        return sample_buffer;
    }
}