#pragma once
#ifndef RTAW_POLICY_H_
#define RTAW_POLICY_H_

#include <torch/torch.h>
#include <torch/script.h>
#include <string>
#include "policy/categorical.h"
#include "policy/buffers.h"
namespace rds
{

    class GNNAllocationPolicy
    {
    public:
        GNNAllocationPolicy() {};
        PolicyOut action_sample(torch::jit::script::Module model, torch::Tensor node_features, torch::Tensor edge_index,
                                torch::Tensor edge_features, torch::Tensor start_ids, torch::Tensor target_ids, 
                                torch::Tensor robot_ids, torch::Tensor selected_ids)
        {
            c10::IValue out = model.forward(torch::jit::Stack({node_features, edge_index, edge_features,
                                                               start_ids.reshape({1, -1}),
                                                               target_ids.reshape({1, -1}),
                                                               robot_ids, selected_ids}));
            torch::Tensor probs = out.toTuple()->elements()[0].toTensor();
            torch::Tensor value = out.toTuple()->elements()[1].toTensor();
            Categorical distribution = Categorical(probs);
            torch::Tensor action = distribution.sample();
            return PolicyOut(action, value, distribution.log_prob(action), distribution.entropy());
        }

        PolicyOut action_evaluation(torch::jit::script::Module model, std::vector<torch::Tensor> node_features,
                                    std::vector<torch::Tensor> edge_index, std::vector<torch::Tensor> edge_features,
                                    std::vector<torch::Tensor> start_ids, std::vector<torch::Tensor> target_ids,
                                    std::vector<torch::Tensor> robot_ids, std::vector<torch::Tensor> selected_ids,
                                    torch::Tensor action)
        {
            torch::Tensor node_feats = node_features[0];
            torch::Tensor edge_feats = edge_features[0];
            torch::Tensor edge_indices = edge_index[0];
            torch::Tensor ids_start = start_ids[0].reshape({1, -1});
            torch::Tensor ids_target = target_ids[0].reshape({1, -1});
            torch::Tensor ids_robot = robot_ids[0];
            torch::Tensor ids_selected = selected_ids[0];
            for (int i = 1; i < node_features.size(); i++)
            {
                edge_indices = torch::cat({edge_indices, edge_index[i] + node_feats.size(0)}, 1);
                node_feats = torch::cat({node_feats, node_features[i]});
                edge_feats = torch::cat({edge_feats, edge_features[i]});
                ids_robot = torch::cat({ids_robot, robot_ids[i]});
                ids_selected = torch::cat({ids_selected, selected_ids[i]});
                ids_start = torch::cat({ids_start, start_ids[i].reshape({1, -1})});
                ids_target = torch::cat({ids_target, target_ids[i].reshape({1, -1})});
            }
            c10::IValue out = model.forward(torch::jit::Stack({node_feats, edge_indices, edge_feats, ids_start, 
                                                                ids_target, ids_robot, ids_selected}));
            torch::Tensor probs = out.toTuple()->elements()[0].toTensor();
            torch::Tensor value = out.toTuple()->elements()[1].toTensor();
            Categorical distribution = Categorical(probs);

            return PolicyOut(action, value, distribution.log_prob(action), distribution.entropy());
        }

        torch::Tensor action_deterministic(torch::jit::script::Module model, torch::Tensor node_features, torch::Tensor edge_index,
                                           torch::Tensor edge_features, torch::Tensor start_ids, torch::Tensor target_ids, 
                                           torch::Tensor robot_ids, torch::Tensor selected_ids)
        {
            c10::IValue out = model.forward(torch::jit::Stack({node_features, edge_index, edge_features,
                                                               start_ids.reshape({1, -1}),
                                                               target_ids.reshape({1, -1}),
                                                               robot_ids, selected_ids}));
            torch::Tensor probs = out.toTuple()->elements()[0].toTensor();
            return probs.argmax(1);
        }

        // torch::Tensor value(torch::jit::script::Module model, torch::Tensor node_features, torch::Tensor edge_index,
        //                     torch::Tensor edge_features, torch::Tensor start_ids, torch::Tensor target_ids)
        // {
        //     c10::IValue out = model.forward(torch::jit::Stack({node_features, edge_index, edge_features, start_ids, target_ids}));
        //     return out.toTuple()->elements()[1].toTensor();
        // }
    };

    struct ObservationFromBuffer
    {
        std::vector<torch::Tensor> node_features, edge_features, edge_index, start_ids, target_ids, robot_ids, selected_ids;
        ObservationFromBuffer(std::vector<torch::Tensor> node_feats, std::vector<torch::Tensor> edge_idx,
                              std::vector<torch::Tensor> edge_feats, std::vector<torch::Tensor> ids_start,
                              std::vector<torch::Tensor> ids_target, std::vector<torch::Tensor> ids_robot,
                              std::vector<torch::Tensor> ids_selected)
        {
            node_features = node_feats;
            edge_index = edge_idx;
            edge_features = edge_feats;
            start_ids = ids_start;
            target_ids = ids_target;
            robot_ids = ids_robot;
            selected_ids = ids_selected;
        }

        ObservationFromBuffer()
        {
            node_features.clear();
            edge_features.clear();
            edge_index.clear();
            start_ids.clear();
            target_ids.clear();
            robot_ids.clear();
            selected_ids.clear();
        }
    };

    ObservationFromBuffer get_observation_from_buffer(GNNAllocationBuffers buffers, const std::vector<int> indices)
    {
        ObservationFromBuffer observation;
        for (int id : indices)
        {
            observation.node_features.push_back(buffers.node_features[id]);
            observation.edge_features.push_back(buffers.edge_features[id]);
            observation.edge_index.push_back(buffers.edge_indices[id]);
            observation.start_ids.push_back(buffers.start_ids[id]);
            observation.target_ids.push_back(buffers.target_ids[id]);
            observation.robot_ids.push_back(buffers.robot_ids[id]);
            observation.selected_ids.push_back(buffers.selected_ids[id]);
        }

        return observation;
    }
}
#endif