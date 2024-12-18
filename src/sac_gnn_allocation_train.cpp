#include <iostream>
#include <vector>
#include <torch/torch.h>
int main()
{
    // torch::Tensor x = torch::randn(10);
    // std::cout << x << std::endl;
    // std::vector<torch::Tensor> action;
    // for (int i = 0; i < x.size(0); i++)
    // {
    //     action.push_back(x[i]);
    // }
    // torch::Tensor y = action.front().reshape(1);
    // std::cout << y.sizes() << std::endl;
    // for (int i = 1; i < action.size(); i++)
    // {
    //     y = torch::cat({y, action[i].reshape(1)}, 0);
    // }
    // std::cout << y << std::endl;
    torch::Tensor sample_ids_torch = torch::randint(0, 10, 5).toType(torch::kInt32);
    std::vector<int> sample_ids(sample_ids_torch.data_ptr<int>(), sample_ids_torch.data_ptr<int>() + sample_ids_torch.numel());
    std::cout << sample_ids << std::endl;
    
    return 0;
}