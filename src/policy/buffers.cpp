#include "policy/buffers.h"

namespace rds
{
    RTAWBuffer::RTAWBuffer(int batch_size, int num_robots, int num_tasks, float gamma, float lambda, torch::DeviceType device)
    {
        batch_size_ = batch_size;
        num_robots_ = num_robots;
        num_tasks_ = num_tasks;
        gamma_ = gamma;
        lambda_ = lambda;
        counter_ = 0;
        has_last_value_ = false;
        device_ = device;
        for (int i = 0; i < batch_size_; ++i)
            reward_states_.push_back(false);
        {
            torch::NoGradGuard no_grad;
            selects_ = torch::zeros({batch_size_, 3}).to(device_);
            robots_ = torch::zeros({batch_size_, num_robots_, 3}).to(device_);
            tasks_ = torch::zeros({batch_size_, num_tasks_, 6}).to(device_);
            actions_ = torch::zeros(batch_size_).to(device_);
            rewards_ = torch::zeros(batch_size_).to(device_);
            values_ = torch::zeros(batch_size_).to(device_);
            log_probs_ = torch::zeros(batch_size_).to(device_);
            advantages_ = torch::zeros(batch_size_).to(device_);
            returns_ = torch::zeros(batch_size_).to(device_);
            last_value_ = torch::zeros(1).to(device_);
        }
    }

    void RTAWBuffer::update(torch::Tensor select, torch::Tensor robot, torch::Tensor task, torch::Tensor action, torch::Tensor value, torch::Tensor log_prob)
    {
        {
            torch::NoGradGuard no_grad;
            selects_[counter_] = select[0];
            robots_[counter_] = robot[0];
            tasks_[counter_] = task[0];
            actions_[counter_] = action[0];
            values_[counter_] = value[0];
            log_probs_[counter_] = log_prob[0];
            counter_ += 1;
        }
    }

    void RTAWBuffer::estimate_advantage()
    {
        {
            torch::NoGradGuard no_grad;
            float last_gae = 0.0f, next_value = 0.0f, delta = 0.0f;
            for (int idx = batch_size_ - 1; idx >= 0; idx--)
            {
                if (idx == batch_size_ - 1)
                {
                    next_value = last_value_[0].cpu().detach().item().toFloat();
                }
                else
                {
                    next_value = values_[idx + 1].cpu().detach().item().toFloat();
                }
                delta = rewards_[idx].cpu().detach().item().toFloat() + gamma_ * next_value - values_[idx].cpu().detach().item().toFloat();
                advantages_[idx] = delta + gamma_ * lambda_ * last_gae;
                last_gae = delta + gamma_ * lambda_ * last_gae;
            }
            returns_ = advantages_ + values_;
        }
    }

    void RTAWBuffer::clear()
    {
        counter_ = 0;
        has_last_value_ = false;

        for (int i = 0; i < batch_size_; i++)
        {
            reward_states_[i] = false;
        }
        {
            torch::NoGradGuard no_grad;
            selects_ = torch::zeros({batch_size_, 3}).to(device_);
            robots_ = torch::zeros({batch_size_, num_robots_, 3}).to(device_);
            tasks_ = torch::zeros({batch_size_, num_tasks_, 6}).to(device_);
            actions_ = torch::zeros(batch_size_).to(device_);
            rewards_ = torch::zeros(batch_size_).to(device_);
            values_ = torch::zeros(batch_size_).to(device_);
            log_probs_ = torch::zeros(batch_size_).to(device_);
            advantages_ = torch::zeros(batch_size_).to(device_);
            returns_ = torch::zeros(batch_size_).to(device_);
            last_value_ = torch::zeros(1).to(device_);
        }
    }

    GNNAllocationBuffer::GNNAllocationBuffer(int batch_size, float gamma, float lambda, torch::DeviceType device)
    {
        batch_size_ = batch_size;
        gamma_ = gamma;
        lambda_ = lambda;
        counter_ = 0;
        has_last_value_ = false;
        device_ = device;
        for (int i = 0; i < batch_size_; ++i)
            reward_states_.push_back(false);
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
            values_ = torch::zeros(batch_size_).to(device_);
            log_probs_ = torch::zeros(batch_size_).to(device_);
            advantages_ = torch::zeros(batch_size_).to(device_);
            returns_ = torch::zeros(batch_size_).to(device_);
            last_value_ = torch::zeros(1).to(device_);
        }
    }

    void GNNAllocationBuffer::update(torch::Tensor node_feats, torch::Tensor edge_feats, torch::Tensor edge_index, 
                                    torch::Tensor start_ids, torch::Tensor target_ids, torch::Tensor robot_ids, 
                                    torch::Tensor selected_ids, torch::Tensor action, torch::Tensor value, torch::Tensor log_prob)
    {
        {
            torch::NoGradGuard no_grad;
            node_features_.push_back(node_feats);
            edge_features_.push_back(edge_feats);
            edge_indices_.push_back(edge_index);
            start_ids_.push_back(start_ids);
            target_ids_.push_back(target_ids);
            robot_ids_.push_back(robot_ids); 
            selected_ids_.push_back(selected_ids);  
            actions_[counter_] = action[0];
            values_[counter_] = value[0];
            log_probs_[counter_] = log_prob[0];
            counter_ += 1;
        }
    }

    void GNNAllocationBuffer::estimate_advantage()
    {
        {
            torch::NoGradGuard no_grad;
            float last_gae = 0.0f, next_value = 0.0f, delta = 0.0f;
            for (int idx = batch_size_ - 1; idx >= 0; idx--)
            {
                if (idx == batch_size_ - 1)
                {
                    next_value = last_value_[0].cpu().detach().item().toFloat();
                }
                else
                {
                    next_value = values_[idx + 1].cpu().detach().item().toFloat();
                }
                delta = rewards_[idx].cpu().detach().item().toFloat() + gamma_ * next_value - values_[idx].cpu().detach().item().toFloat();
                advantages_[idx] = delta + gamma_ * lambda_ * last_gae;
                last_gae = delta + gamma_ * lambda_ * last_gae;
            }
            returns_ = advantages_ + values_;
        }
    }

    void GNNAllocationBuffer::clear()
    {
        counter_ = 0;
        has_last_value_ = false;

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
            values_ = torch::zeros(batch_size_).to(device_);
            log_probs_ = torch::zeros(batch_size_).to(device_);
            advantages_ = torch::zeros(batch_size_).to(device_);
            returns_ = torch::zeros(batch_size_).to(device_);
            last_value_ = torch::zeros(1).to(device_);
        }
    }

}