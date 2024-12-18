#ifndef CATEGORICAL_H
#define CATEGORICAL_H

#include <torch/torch.h>

namespace rds
{
    struct PolicyOut
    {
        torch::Tensor action, log_prob, value, entropy;
        PolicyOut(torch::Tensor action_, torch::Tensor value_, torch::Tensor log_prob_, torch::Tensor entropy_)
        {
            action = action_;
            value = value_;
            log_prob = log_prob_;
            entropy = entropy_;
        }
    };
    typedef std::shared_ptr<PolicyOut> PolicyOutHandle;
    
    class Categorical
    {
    public:
        Categorical(torch::Tensor probs);
        ~Categorical();

        std::vector<int64_t> extended_shape(torch::ArrayRef<int64_t> shape);
        torch::Tensor sample(torch::ArrayRef<int64_t> shape = {});
        torch::Tensor entropy();
        torch::Tensor log_prob(torch::Tensor act_prob);

    private:
        int m_num_events;
        std::vector<int64_t> m_batch_shape;
        torch::Tensor m_probs, m_logits;
    };
}
#endif