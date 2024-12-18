#include "policy/categorical.h"
namespace rds
{
	Categorical::Categorical(torch::Tensor probs)
	{
		torch::Tensor param;

		m_batch_shape = {};
		m_probs = probs / probs.sum(-1, true);
		m_probs = this->m_probs.clamp(1.21e-7, 1.0 - 1.21e-7);
		m_logits = torch::log(this->m_probs);

		param = probs;
		m_num_events = param.size(-1);

		if (param.dim() > 1)
		{
			m_batch_shape = param.sizes().vec();
			m_batch_shape.resize(m_batch_shape.size() - 1);
		}
	}

	Categorical::~Categorical()
	{
	}

	// refer to /usr/lib/python3.7/site-packages/torch/distributions/distribution.py
	std::vector<int64_t> Categorical::extended_shape(c10::ArrayRef<int64_t> shape)
	{
		std::vector<int64_t> output_shape;

		output_shape.insert(output_shape.end(), shape.begin(), shape.end());
		output_shape.insert(output_shape.end(), m_batch_shape.begin(), m_batch_shape.end());
		// output_shape.insert(output_shape.end(),	m_event_shape.begin(), m_event_shape.end());

		return output_shape;
	}

	// refer to /usr/lib/python3.7/site-packages/torch/distributions/categorical.py
	torch::Tensor Categorical::sample(torch::ArrayRef<int64_t> shape)
	{
		std::vector<int64_t> vec_shape;
		std::vector<int64_t> param_shape;
		torch::Tensor exp_probs;
		torch::Tensor probs_2d;
		torch::Tensor sample_2d;

		vec_shape = extended_shape(shape);
		param_shape = vec_shape;
		param_shape.insert(param_shape.end(), {m_num_events});

		exp_probs = m_probs.expand(param_shape);

		if (m_probs.dim() == 1 || m_probs.size(0) == 1)
			probs_2d = exp_probs.view({-1, m_num_events});
		else
			probs_2d = exp_probs.contiguous().view({-1, m_num_events});

		sample_2d = torch::multinomial(probs_2d, 1, true);

		return sample_2d.contiguous().view(vec_shape); // out is int64_t
	}

	torch::Tensor Categorical::entropy()
	{
		torch::Tensor p_log_p = m_logits * m_probs;
		return -p_log_p.sum(-1).to(torch::kFloat64);
	}

	torch::Tensor Categorical::log_prob(torch::Tensor act_prob)
	{
		std::vector<torch::Tensor> broadcasted_tensors;

		act_prob = act_prob.to(torch::kInt64).unsqueeze(-1);
		broadcasted_tensors = torch::broadcast_tensors({act_prob, m_logits});
		act_prob = broadcasted_tensors[0];
		act_prob = act_prob.narrow(-1, 0, 1);

		return broadcasted_tensors[1].gather(-1, act_prob).squeeze(-1).to(torch::kFloat64);
	}
}