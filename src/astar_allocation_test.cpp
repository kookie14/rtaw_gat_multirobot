#include <iostream>
#include "env/single_environment.h"
#include "policy/buffers.h"
#include "policy/rtaw_policy.h"
#include "planner/astar_planner.h"
#include "allocation/task_generator.h"
#include "allocation/astar_allocation.h"
#include "opencv4/opencv2/opencv.hpp"
#include <chrono>
#include <thread>
#include <torch/script.h>
#include <filesystem>
#include <numeric>
#include <bits/stdc++.h>

int main()
{
    int seed_value = 1;
    int num_test_robots = 50;
    int num_test_task = 500;
    int num_map_rows = 5;
    int num_map_cols = 5;
    std::string env_name;
    std::cout << "Enter name of environment: ";
    std::cin >> env_name;
    std::cout << "Enter seed value: ";
    std::cin >>  seed_value;
    std::cout << "Enter num test robots: ";
    std::cin >> num_test_robots;
    std::cout << "Enter num test tasks: ";
    std::cin >> num_test_task;
    std::cout << "Enter num map rows: ";
    std::cin >> num_map_rows;
    std::cout << "Enter num map cols: ";
    std::cin >> num_map_cols;

    torch::manual_seed(seed_value);
    auto cuda_available = torch::cuda::is_available();
    torch::DeviceType device(cuda_available ? torch::kCUDA : torch::kCPU);
    std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';
    std::string env_data_path = "../data/layout/single/"+ env_name;
    std::string model_folder = "../data/trained_model/single";
    float robot_max_speed = 1.0;
    float robot_max_payload = 200.0;
    float waiting_time = 3.0;
    int num_task_queue = 16;
    int num_priority = 5;
    int num_type = 2;
    float min_mass = 10.0f;
    float max_mass = 200.0f;

    float x_visual = 38;   // 38, 33
    float y_visual = 32.5; // 32.5, 52.5
    float font_size_visual = 25.0f;
    float scale_figure = 25.0f;

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

    rds::TaskGeneratorHandle test_task_gen = std::make_shared<rds::TaskGenerator>(num_task_queue, num_priority, num_type, min_mass,
                                                                                  max_mass, test_env->graph_handle(), rds::RANDOM_MODE, env_data_path);
    rds::RTAWBufferHandle test_buffer = std::make_shared<rds::RTAWBuffer>(num_test_task, num_test_robots, num_task_queue, 0.99, 0.95, device);
    rds::AStarAllocationHandle test_allocation = std::make_shared<rds::AStarAllocation>(test_env->swarm(), test_env->graph_handle(),
                                                                                        test_task_gen, test_buffer);
    rds::PlannerInterfaceHandle test_planner = std::make_shared<rds::AStarPlanner>(
        test_env->swarm(), test_env->graph_handle(), test_env->map_center(), test_env->map_length(),
        test_env->map_width(), test_env->single_line_width(), test_env->double_line_width());

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
        rds::AStarObservation obs = test_allocation->observation();
        if (obs.state == true)
        {
            int action = std::distance(obs.distances.begin(), std::min_element(obs.distances.begin(), obs.distances.end()));
            // std::cout << action << std::endl;
            test_allocation->allocation(obs.robot_id, action);
            if (test_buffer->is_full() == false)
            {
                test_buffer->update(torch::zeros({1, 3}).to(device), torch::zeros({1, num_test_robots, 3}).to(device),
                                    torch::zeros({1, num_task_queue, 6}).to(device), torch::zeros(1), torch::zeros(1),
                                    torch::zeros(1));
            }
            if (test_buffer->is_full() == true && test_buffer->has_last_value() == false)
            {
                test_buffer->set_has_last_value(true);
                test_buffer->set_last_value(torch::zeros(1));
            }
        }
        test_planner->planning();
        test_env->controller(waiting_time);
        test_allocation->update_reward();
        test_env->visualize();
        if (i == std::max(num_test_robots, 100))
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
            }
            std::vector<bool> task_state = test_buffer->reward_states();
            int num_task_done = std::count(task_state.begin(), task_state.end(), true);
            test_completed_tasks_visuals->label_values({"Completed tasks : " + std::to_string(num_task_done) + "/" + std::to_string(task_state.size())});
            for (int robot_id = 0; robot_id < test_env->swarm().size(); robot_id++)
            {
                prev_test_positions[robot_id].set(test_env->swarm()[robot_id]->x(),
                                                  test_env->swarm()[robot_id]->y());
            }
            fig_test->save("../visualize/env_astar_test.png");
        }
        i += 1;
        if (test_buffer->is_complete())
            break;
    }
    std::cout << "Sum completed time: " << test_buffer->rewards().sum().item().toFloat() << " s." << std::endl;

    return 0;
}