#pragma once
#ifndef RTAW_POLICY_H_
#define RTAW_POLICY_H_

#include <torch/torch.h>
#include <torch/script.h>
#include <string>
#include "policy/categorical.h"

namespace rds
{
    class RTAWPolicy
    {
    public:
        RTAWPolicy(){};
        PolicyOut action_sample(torch::jit::script::Module model, torch::Tensor selected, torch::Tensor robot, torch::Tensor task)
        {
            c10::IValue out = model.forward(torch::jit::Stack({selected, robot, task}));
            torch::Tensor probs = out.toTuple()->elements()[0].toTensor();
            torch::Tensor value = out.toTuple()->elements()[1].toTensor();
            Categorical distribution = Categorical(probs);
            torch::Tensor action = distribution.sample();
            return PolicyOut(action, value, distribution.log_prob(action), distribution.entropy());
        }

        PolicyOut action_evaluation(torch::jit::script::Module model, torch::Tensor selected, torch::Tensor robot, torch::Tensor task, torch::Tensor action)
        {
            c10::IValue out = model.forward(torch::jit::Stack({selected, robot, task}));
            torch::Tensor probs = out.toTuple()->elements()[0].toTensor();
            torch::Tensor value = out.toTuple()->elements()[1].toTensor();
            Categorical distribution = Categorical(probs);
            return PolicyOut(action, value, distribution.log_prob(action), distribution.entropy());
        }

        torch::Tensor action_deterministic(torch::jit::script::Module model, torch::Tensor selected, torch::Tensor robot, torch::Tensor task)
        {
            c10::IValue out = model.forward(torch::jit::Stack({selected, robot, task}));
            torch::Tensor probs = out.toTuple()->elements()[0].toTensor();
            return probs.argmax(1);
        }

        torch::Tensor value(torch::jit::script::Module model, torch::Tensor selected, torch::Tensor robot, torch::Tensor task)
        {
            c10::IValue out = model.forward(torch::jit::Stack({selected, robot, task}));
            return out.toTuple()->elements()[1].toTensor();
        }
    };
}
#endif