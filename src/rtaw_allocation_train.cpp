#include <iostream>
#include "env/single_environment.h"
#include "policy/buffers.h"
#include "policy/rtaw_policy.h"
#include "planner/astar_planner.h"
#include "allocation/task_generator.h"
#include "allocation/rtaw_allocation.h"
#include "opencv4/opencv2/opencv.hpp"
#include <chrono>
#include <thread>
#include <torch/script.h>
#include <filesystem>
#include <numeric>
#include <bits/stdc++.h>

void save_model(std::string folder_path, torch::jit::script::Module module, int iteration)
{
    module.save(folder_path + "/model.pth");
    std::ofstream iteration_file;
    iteration_file.open(folder_path + "/iteration.txt", std::ios_base::out);
    iteration_file << iteration;
    iteration_file.close();
}

void save_best_average_time_model(std::string folder_path, torch::jit::script::Module module, float best_average_time)
{
    module.save(folder_path + "/best_average_time_model.pth");
    std::ofstream best_average_time_file;
    best_average_time_file.open(folder_path + "/best_average_time.txt", std::ios_base::out);
    best_average_time_file << best_average_time;
    best_average_time_file.close();
}

void save_best_execution_time_model(std::string folder_path, torch::jit::script::Module module, float best_execution_time)
{
    module.save(folder_path + "/best_execution_time_model.pth");
    std::ofstream best_execution_time_file;
    best_execution_time_file.open(folder_path + "/best_execution_time.txt", std::ios_base::out);
    best_execution_time_file << best_execution_time;
    best_execution_time_file.close();
}

void save_best_reward(std::string folder_path, float best_reward)
{
    std::ofstream best_reward_file;
    best_reward_file.open(folder_path + "/best_reward.txt", std::ios_base::out);
    best_reward_file << best_reward;
    best_reward_file.close();
}

int load_iteration(std::string folder_path)
{
    if (std::filesystem::exists(folder_path + "/iteration.txt"))
    {
        std::ifstream file(folder_path + "/iteration.txt");
        int iteration;
        file >> iteration;
        file.close();
        return iteration + 1;
    }
    return 1;
}

float load_best_average_time(std::string folder_path)
{
    if (std::filesystem::exists(folder_path + "/best_average_time.txt"))
    {
        std::ifstream file(folder_path + "/best_average_time.txt");
        float best_average_time;
        file >> best_average_time;
        file.close();
        std::cout << "Best average time: " << best_average_time << " s" << std::endl;
        return best_average_time;
    }
    return std::numeric_limits<float>::infinity();
}

float load_best_execution_time(std::string folder_path)
{
    if (std::filesystem::exists(folder_path + "/best_execution_time.txt"))
    {
        std::ifstream file(folder_path + "/best_execution_time.txt");
        float best_execution_time;
        file >> best_execution_time;
        file.close();
        std::cout << "Best execution time: " << best_execution_time << " s" << std::endl;
        return best_execution_time;
    }
    return std::numeric_limits<float>::infinity();
}

float load_best_reward(std::string folder_path)
{
    if (std::filesystem::exists(folder_path + "/best_reward.txt"))
    {
        std::ifstream file(folder_path + "/best_reward.txt");
        float best_reward;
        file >> best_reward;
        file.close();
        std::cout << "Best reward: " << best_reward << std::endl;
        return best_reward;
    }
    return -std::numeric_limits<float>::infinity();
}

void logger(std::string folder_path, int iter, float reward, float learning_rate, float policy_loss, float value_loss, float entropy_loss)
{
    if (std::filesystem::exists(folder_path + "/logger.csv"))
    {
        std::ofstream logger_file;
        logger_file.open(folder_path + "/logger.csv", std::ios_base::app);
        logger_file << iter << "," << reward << "," << learning_rate << "," << policy_loss << "," << value_loss << "," << entropy_loss << std::endl;
        logger_file.close();
    }
    else
    {
        std::ofstream logger_file;
        logger_file.open(folder_path + "/logger.csv", std::ios_base::out);
        logger_file << "iteration,rewards,learning rate,policy_loss,value_loss,entropy_loss" << std::endl;
        logger_file << iter << "," << reward << "," << learning_rate << "," << policy_loss << "," << value_loss << "," << entropy_loss << std::endl;
        logger_file.close();
    }
}

int main()
{
    torch::manual_seed(5);
    auto cuda_available = torch::cuda::is_available();
    torch::DeviceType device(cuda_available ? torch::kCUDA : torch::kCPU);
    std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';
    std::string env_data_path = "../data/layout/single/100x50";
    std::string model_folder = "../data/trained_model";
    int num_train_envs = 8;
    int num_train_robots = 32;
    int num_test_robots = 32;
    int num_train_task = 64;
    int num_test_task = 100;
    float robot_max_speed = 1.0;
    float robot_max_payload = 200.0;
    float waiting_time = 3.0;
    int num_map_rows = 5;
    int num_map_cols = 5;
    int num_task_queue = 16;
    int num_priority = 5;
    int num_type = 2;
    float min_mass = 10.0f;
    float max_mass = 200.0f;
    // PPO parameters
    int mini_batch_size = 32;
    float lr = 3e-4;
    int num_epochs = 16;
    float gamma = 0.99;
    float lambda = 0.95;
    float entropy_coeff = 0.01;
    float value_coeff = 0.0002;
    float clip_coeff = 0.2;
    bool clip_vloss = true;
    float max_grad_norm = 0.5;
    float target_kl = 0.01;
    int num_iterations = 100;
    bool norm_adv = true;
    bool anneal_lr = true;

    // Train environments
    float x_visual = 83; // 38, 33
    float y_visual = 52.5; // 32.5
    float font_size_visual = 25.0f;
    float scale_figure = 25.0f;
    std::vector<matplot::figure_handle> train_figs;
    std::vector<matplot::axes_handle> train_axes;
    std::vector<rds::EnvironmentInterfaceHandle> train_envs;
    std::vector<rds::TaskGeneratorHandle> train_task_gens;
    std::vector<rds::PlannerInterfaceHandle> train_planners;
    std::vector<matplot::labels_handle> completed_tasks_visuals;
    std::vector<rds::RTAWBufferHandle> train_buffers;
    std::vector<rds::RTAWAllocationHandle> train_allocations;
    std::vector<std::vector<rds::Point2D>> prev_positions;
    for (int id = 0; id < num_train_envs; id++)
    {
        matplot::figure_handle fig = matplot::figure(true);
        matplot::axes_handle ax = fig->current_axes();
        train_envs.push_back(
            std::make_shared<rds::SingleEnvironment>(
                rds::RANDOM_INITIALIZATION, env_data_path, num_train_robots, robot_max_speed,
                robot_max_payload, ax, num_map_rows, num_map_cols, waiting_time));
        train_envs[id]->visualize();

        completed_tasks_visuals.push_back(ax->text(x_visual, y_visual, "Completed tasks: 0/0"));
        completed_tasks_visuals[id]->color(rds::RED);
        completed_tasks_visuals[id]->font_size(font_size_visual);

        ax->axes_aspect_ratio(train_envs[id]->map_width() / train_envs[id]->map_length());
        ax->position({0.0f, 0.0f, 1.0f, 1.0f});
        fig->size(int(train_envs[id]->map_length()) * scale_figure, int(train_envs[id]->map_width()) * scale_figure);
        ax->xticks({});
        ax->yticks({});

        train_task_gens.push_back(std::make_shared<rds::TaskGenerator>(num_task_queue, num_priority, num_type, min_mass,
                                                                       max_mass, train_envs[id]->graph_handle(), rds::RANDOM_MODE));
        train_buffers.push_back(std::make_shared<rds::RTAWBuffer>(num_train_task, num_train_robots, num_task_queue, gamma, lambda, device));
        train_allocations.push_back(std::make_shared<rds::RTAWAllocation>(train_envs[id]->swarm(), train_envs[id]->graph_handle(),
                                                                          train_task_gens[id], train_buffers[id], train_envs[id]->map_length(),
                                                                          train_envs[id]->map_width(), device, train_envs[id]->dt(), rds::TRAINING));
        train_planners.push_back(
            std::make_shared<rds::AStarPlanner>(train_envs[id]->swarm(), train_envs[id]->graph_handle(), train_envs[id]->map_center(),
                                                train_envs[id]->map_length(), train_envs[id]->map_width(),
                                                train_envs[id]->single_line_width(), train_envs[id]->double_line_width()));

        train_figs.push_back(fig);
        train_axes.push_back(ax);

        std::vector<rds::Point2D> prev_position;
        for (rds::RobotHandle robot : train_envs[id]->swarm())
        {
            prev_position.push_back(robot->position());
        }
        prev_positions.push_back(prev_position);
    }

    // Test environment
    matplot::figure_handle fig_test = matplot::figure(true);
    matplot::axes_handle ax_test = fig_test->current_axes();
    rds::EnvironmentInterfaceHandle test_env = std::make_shared<rds::SingleEnvironment>(
        rds::TEST_INITIALIZATION, env_data_path, num_test_robots, robot_max_speed,
        robot_max_payload, ax_test, num_map_rows, num_map_cols, waiting_time);
    ax_test->axes_aspect_ratio(test_env->map_width() / test_env->map_length());
    ax_test->position({0.0f, 0.0f, 1.0f, 1.0f});
    fig_test->size(int(test_env->map_length()) * scale_figure, int(test_env->map_width()) * scale_figure);
    ax_test->xticks({});
    ax_test->yticks({});
    matplot::labels_handle test_completed_tasks_visuals = ax_test->text(x_visual, y_visual, "Completed tasks: 0/0");
    test_completed_tasks_visuals->color(rds::RED);
    test_completed_tasks_visuals->font_size(font_size_visual);
    ;
    rds::TaskGeneratorHandle test_task_gen = std::make_shared<rds::TaskGenerator>(num_task_queue, num_priority, num_type, min_mass,
                                                                                  max_mass, test_env->graph_handle(), rds::RANDOM_MODE, env_data_path);
    rds::RTAWBufferHandle test_buffer = std::make_shared<rds::RTAWBuffer>(num_test_task, num_test_robots, num_task_queue, gamma, lambda, device);
    rds::RTAWAllocationHandle test_allocation = std::make_shared<rds::RTAWAllocation>(test_env->swarm(), test_env->graph_handle(),
                                                                                      test_task_gen, test_buffer, test_env->map_length(),
                                                                                      test_env->map_width(), device, test_env->dt(), rds::TESTING);
    rds::PlannerInterfaceHandle test_planner = std::make_shared<rds::AStarPlanner>(
        test_env->swarm(), test_env->graph_handle(), test_env->map_center(), test_env->map_length(),
        test_env->map_width(), test_env->single_line_width(), test_env->double_line_width());

    torch::jit::script::Module rtaw_module;
    if (std::filesystem::exists(model_folder + "/model.pth"))
    {
        rtaw_module = torch::jit::load(model_folder + "/model.pth", device);
    }
    else
    {
        rtaw_module = torch::jit::load("../data/original_model/rtaw/rtaw_policy4.pth", device);
    }

    rtaw_module.train(true);
    std::vector<at::Tensor> parameters;
    for (const auto &params : rtaw_module.parameters())
    {
        parameters.push_back(params);
    }

    torch::optim::Adam optimizer(parameters, torch::optim::AdamOptions(lr).eps(1e-5));
    if (std::filesystem::exists(model_folder + "/optimizer.pth"))
    {
        std::ifstream optimizer_state_file(model_folder + "/optimizer.pth", std::ios::binary);
        torch::serialize::InputArchive archive;
        archive.load_from(optimizer_state_file);
        optimizer.load(archive);

        optimizer_state_file.close();
    }
    rds::RTAWPolicy policy = rds::RTAWPolicy();

    int start_iteration = load_iteration(model_folder);
    float best_average_time = load_best_average_time(model_folder);
    float best_execution_time = load_best_execution_time(model_folder);
    float best_reward = load_best_reward(model_folder);

    auto rng = std::default_random_engine{};
    for (int iter = start_iteration; iter <= num_iterations; iter++)
    {
        int i = 0;
        while (true)
        {
            int collect_buffer_done_counter = 0;
            for (int env_id = 0; env_id < num_train_envs; env_id++)
            {
                if (train_buffers[env_id]->is_complete())
                {
                    collect_buffer_done_counter += 1;
                    continue;
                }
                train_envs[env_id]->add_all_robot_graph();
                rds::RTAWObservation obs = train_allocations[env_id]->observation();
                if (obs.state == true)
                {
                    rds::PolicyOut out = policy.action_sample(rtaw_module, obs.select, obs.robot, obs.task);
                    train_allocations[env_id]->allocation(obs.robot_id, out.action.item().toInt());
                    if (train_buffers[env_id]->is_full() == false)
                    {
                        train_buffers[env_id]->update(obs.select, obs.robot, obs.task, out.action, out.value, out.log_prob);
                    }
                    if (train_buffers[env_id]->is_full() == false && train_buffers[env_id]->has_last_value() == false)
                    {
                        train_buffers[env_id]->set_has_last_value(true);
                        train_buffers[env_id]->set_last_value(out.value);
                    }
                }
                train_planners[env_id]->planning();
                train_envs[env_id]->controller(waiting_time);
                train_allocations[env_id]->update_reward();
                train_envs[env_id]->visualize();
                if (i == 100)
                {
                    int stuck = 0;
                    for (int robot_id = 0; robot_id < train_envs[env_id]->swarm().size(); robot_id++)
                    {
                        if (rds::is_same_point(train_envs[env_id]->swarm()[robot_id]->position(), prev_positions[env_id][robot_id]))
                        {
                            stuck += 1;
                        }
                    }
                    if (stuck > train_envs[env_id]->swarm().size() / 2)
                    {
                        std::cout << "Environment " << env_id << " is stuck. Reset environment!" << std::endl;
                        train_envs[env_id]->reset();
                        train_buffers[env_id]->clear();
                    }
                    std::vector<bool> task_state = train_buffers[env_id]->reward_states();
                    int num_task_done = std::count(task_state.begin(), task_state.end(), true);
                    completed_tasks_visuals[env_id]->label_values({"Completed tasks : " + std::to_string(num_task_done) + "/" + std::to_string(task_state.size())});
                    for (int robot_id = 0; robot_id < train_envs[env_id]->swarm().size(); robot_id++)
                    {
                        prev_positions[env_id][robot_id].set(train_envs[env_id]->swarm()[robot_id]->x(),
                                                             train_envs[env_id]->swarm()[robot_id]->y());
                    }
                    train_figs[env_id]->save("../visualize/env_" + std::to_string(env_id) + ".png");
                }
            }
            if (i == 100)
            {
                i = 0;
            }
            i += 1;
            if (collect_buffer_done_counter == num_train_envs)
            {
                for (int env_id = 0; env_id < num_train_envs; env_id++)
                {
                    train_buffers[env_id]->estimate_advantage();
                }
                break;
            }
        }
        std::cout << "Done collect buffers at iteration " << iter << std::endl;
        // PPO trainer
        // Update learning rate
        float lr_now = lr;
        if (anneal_lr == true)
        {
            float frac = 1.0 - float((iter - 1.0) / num_iterations);
            lr_now = frac * lr;
            optimizer.param_groups().at(0).options().set_lr(lr_now);
        }
        // Convert all buffers to a buffer
        rds::RTAWBuffers buffers(train_buffers);
        // Test if collect best reward
        // float reward = buffers.rewards.sum().item().toFloat();
        // if (best_reward < reward)
        // {
        //     best_reward = reward;
        //     save_best_reward(model_folder, best_reward);
        // Testing
        {
            rtaw_module.train(false);
            test_env->reset();
            test_buffer->clear();
            test_task_gen->reset();
            float execution_time = 0.0f;
            // Collect test buffers
            int i = 0;
            std::vector<rds::Point2D> prev_test_positions;
            for (rds::RobotHandle robot : test_env->swarm())
            {
                prev_test_positions.push_back(robot->position());
            }
            bool stuck_flag = false;
            while (true)
            {
                test_env->add_all_robot_graph();
                rds::RTAWObservation obs = test_allocation->observation();
                if (obs.state == true)
                {
                    torch::Tensor action = policy.action_deterministic(rtaw_module, obs.select, obs.robot, obs.task);
                    test_allocation->allocation(obs.robot_id, action.item().toInt());
                    if (test_buffer->is_full() == false)
                    {
                        test_buffer->update(obs.select, obs.robot, obs.task, action, action, action);
                    }
                    if (test_buffer->is_full() == true && test_buffer->has_last_value() == false)
                    {
                        test_buffer->set_has_last_value(true);
                        test_buffer->set_last_value(action);
                    }
                }
                test_planner->planning();
                test_env->controller(waiting_time);
                test_allocation->update_reward();
                execution_time += test_env->dt();
                test_env->visualize();
                if (i == 100)
                {
                    i = 0;
                    int stuck = 0;
                    for (int robot_id = 0; robot_id < test_env->swarm().size(); robot_id++)
                    {
                        if (rds::is_same_point(test_env->swarm()[robot_id]->position(), prev_test_positions[robot_id]))
                        {
                            stuck += 1;
                        }
                    }
                    if (stuck > test_env->swarm().size() / 2)
                    {
                        std::cout << "Environment test is stuck!" << std::endl;
                        test_env->reset();
                        test_buffer->clear();
                        test_task_gen->reset();
                        execution_time = 0.0f;
                    }
                    std::vector<bool> task_state = test_buffer->reward_states();
                    int num_task_done = std::count(task_state.begin(), task_state.end(), true);
                    test_completed_tasks_visuals->label_values({"Completed tasks : " + std::to_string(num_task_done) + "/" + std::to_string(task_state.size())});
                    for (int robot_id = 0; robot_id < test_env->swarm().size(); robot_id++)
                    {
                        prev_test_positions[robot_id].set(test_env->swarm()[robot_id]->x(),
                                                          test_env->swarm()[robot_id]->y());
                    }
                    fig_test->save("../visualize/env_test.png");
                }
                i += 1;
                if (test_buffer->is_complete())
                    break;
            }
            float average_time = test_buffer->rewards().mean().item().toFloat();
            std::cout << "Average completed time: " << average_time << " s." << std::endl;
            std::cout << "Execution time: " << execution_time << " s." << std::endl;
            if (best_average_time > average_time)
            {
                best_average_time = average_time;
                save_best_average_time_model(model_folder, rtaw_module, best_average_time);
                std::cout << "Best average completed time: " << best_average_time << " s." << std::endl;
            }
            if (best_execution_time > execution_time)
            {
                best_execution_time = execution_time;
                save_best_execution_time_model(model_folder, rtaw_module, best_execution_time);
                std::cout << "Best execute time: " << best_execution_time << " s." << std::endl;
            }
        }
        // }
        rtaw_module.train(true);
        // Optimize the policy and value network
        int total_batch_size = num_train_envs * num_train_task;
        std::vector<int> b_inds(total_batch_size);
        std::iota(b_inds.begin(), b_inds.end(), 0);
        torch::Tensor policy_loss, v_loss, entropy_loss;
        for (int epoch = 0; epoch < num_epochs; epoch++)
        {
            std::shuffle(b_inds.begin(), b_inds.end(), rng);
            float approx_kl;
            for (int start = 0; start < total_batch_size; start += mini_batch_size)
            {
                int end = start + mini_batch_size;
                std::vector<int> mb_inds = {b_inds.begin() + start, b_inds.begin() + end};
                torch::Tensor mb_inds_tensor = torch::zeros(mb_inds.size()).toType(torch::kInt).to(device);
                for (int idx = 0; idx < mb_inds.size(); idx++)
                    mb_inds_tensor[idx] = mb_inds[idx];
                rds::PolicyOut new_out = policy.action_evaluation(rtaw_module,
                                                                  buffers.selects.index_select(0, mb_inds_tensor),
                                                                  buffers.robots.index_select(0, mb_inds_tensor),
                                                                  buffers.tasks.index_select(0, mb_inds_tensor),
                                                                  buffers.actions.index_select(0, mb_inds_tensor));
                torch::Tensor log_ratio = new_out.log_prob - buffers.log_probs.index_select(0, mb_inds_tensor);
                torch::Tensor ratio = log_ratio.exp();
                {
                    torch::NoGradGuard no_grad;
                    approx_kl = ((ratio - 1) - log_ratio).mean().item().toFloat();
                }
                torch::Tensor mb_advs = buffers.advantages.index_select(0, mb_inds_tensor);
                if (norm_adv)
                {
                    mb_advs = (mb_advs - mb_advs.mean()) / (mb_advs.std() + 1e-8);
                }
                // Policy loss
                torch::Tensor pg_loss1 = -mb_advs * ratio;
                torch::Tensor pg_loss2 = -mb_advs * torch::clamp(ratio, 1 - clip_coeff, 1 + clip_coeff);
                policy_loss = torch::max(pg_loss1, pg_loss2).mean();
                // Value loss

                if (clip_vloss)
                {
                    torch::Tensor v_loss_unclipped = torch::pow((new_out.value - buffers.returns.index_select(0, mb_inds_tensor)), 2);
                    torch::Tensor v_clipped = buffers.values.index_select(0, mb_inds_tensor) + torch::clamp(new_out.value -
                                                                                                                buffers.values.index_select(0, mb_inds_tensor),
                                                                                                            -clip_coeff, clip_coeff);
                    torch::Tensor v_loss_clipped = torch::pow((v_clipped - buffers.returns.index_select(0, mb_inds_tensor)), 2);
                    torch::Tensor v_loss_max = torch::max(v_loss_unclipped, v_loss_clipped);
                    v_loss = 0.5 * v_loss_max.mean();
                }
                else
                {
                    v_loss = 0.5 * torch::pow((new_out.value - buffers.returns.index_select(0, mb_inds_tensor)), 2).mean();
                }
                entropy_loss = new_out.entropy.mean();
                torch::Tensor loss = policy_loss - entropy_coeff * entropy_loss + v_loss * value_coeff;

                optimizer.zero_grad();
                loss.backward();
                torch::nn::utils::clip_grad_norm_(parameters, max_grad_norm);
                optimizer.step();
            }
            if (approx_kl > target_kl)
            {
                break;
            }
        }
        logger(model_folder, iter, buffers.rewards.sum(0).item().toFloat(), lr_now, policy_loss.item().toFloat(),
               v_loss.item().toFloat(), entropy_loss.item().toFloat());
        save_model(model_folder, rtaw_module, iter);
        for (int env_id = 0; env_id < num_train_envs; env_id++)
        {
            train_buffers[env_id]->clear();
        }
        // // Testing
        // if (iter % 5 == 0)
        // {
        //     rtaw_module.train(false);
        //     test_env->reset();
        //     test_buffer->clear();
        //     test_task_gen->reset();
        //     float execution_time = 0.0f;
        //     // Collect test buffers
        //     int i = 0;
        //     std::vector<rds::Point2D> prev_test_positions;
        //     for (rds::RobotHandle robot : test_env->swarm())
        //     {
        //         prev_test_positions.push_back(robot->position());
        //     }
        //     bool stuck_flag = false;
        //     while (true)
        //     {
        //         test_env->add_all_robot_graph();
        //         rds::RTAWObservation obs = test_allocation->observation();
        //         if (obs.state == true)
        //         {
        //             torch::Tensor action = policy.action_deterministic(rtaw_module, obs.select, obs.robot, obs.task);
        //             test_allocation->allocation(obs.robot_id, action.item().toInt());
        //             if (test_buffer->is_full() == false)
        //             {
        //                 test_buffer->update(obs.select, obs.robot, obs.task, action, action, action);
        //             }
        //             if (test_buffer->is_full() == true && test_buffer->has_last_value() == false)
        //             {
        //                 test_buffer->set_has_last_value(true);
        //                 test_buffer->set_last_value(action);
        //             }
        //         }
        //         test_planner->planning();
        //         test_env->controller(waiting_time);
        //         test_allocation->update_reward();
        //         execution_time += test_env->dt();
        //         test_env->visualize();
        //         if (i == 100)
        //         {
        //             i = 0;
        //             int stuck = 0;
        //             for (int robot_id = 0; robot_id < test_env->swarm().size(); robot_id++)
        //             {
        //                 if (rds::is_same_point(test_env->swarm()[robot_id]->position(), prev_test_positions[robot_id]))
        //                 {
        //                     stuck += 1;
        //                 }
        //             }
        //             if (stuck > test_env->swarm().size() / 2)
        //             {
        //                 std::cout << "Environment test is stuck!" << std::endl;
        //                 test_env->reset();
        //                 test_buffer->clear();
        //                 test_task_gen->reset();
        //                 execution_time = 0.0f;
        //             }
        //             std::vector<bool> task_state = test_buffer->reward_states();
        //             int num_task_done = std::count(task_state.begin(), task_state.end(), true);
        //             test_completed_tasks_visuals->label_values({"Completed tasks : " + std::to_string(num_task_done) + "/" + std::to_string(task_state.size())});
        //             for (int robot_id = 0; robot_id < test_env->swarm().size(); robot_id++)
        //             {
        //                 prev_test_positions[robot_id].set(test_env->swarm()[robot_id]->x(),
        //                                                   test_env->swarm()[robot_id]->y());
        //             }
        //             fig_test->save("../visualize/env_test.png");
        //         }
        //         i += 1;
        //         if (test_buffer->is_complete())
        //             break;
        //     }
        //     float average_time = test_buffer->rewards().mean().item().toFloat();
        //     std::cout << "Average completed time: " << average_time << " s." << std::endl;
        //     std::cout << "Execution time: " << execution_time << " s." << std::endl;
        //     if (best_average_time > average_time)
        //     {
        //         best_average_time = average_time;
        //         save_best_average_time_model(model_folder, rtaw_module, best_average_time);
        //         std::cout << "Best average completed time: " << best_average_time << " s." << std::endl;
        //     }
        //     if (best_execution_time > execution_time)
        //     {
        //         best_execution_time = execution_time;
        //         save_best_execution_time_model(model_folder, rtaw_module, best_execution_time);
        //         std::cout << "Best execute time: " << best_execution_time << " s." << std::endl;
        //     }
        // }
    }
    return 0;
}
