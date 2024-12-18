#include "env/real_environment.h"
// #include "env/single_environment.h"
// #include "server/task_generator.h"
// #include "server/astar_allocation.h"
// #include "server/astar_planner_with_waiting.h"
// #include "opencv4/opencv2/opencv.hpp"
// #include "robot/robot.h"
#include <chrono>
#include <thread>
int main()
{
    // matplot::figure_handle fig1 = matplot::figure(true);
    // matplot::axes_handle ax1 = fig1->current_axes();
    // ax1->hold(true);
    // rds::EnvironmentInterfaceHandle env = std::make_shared<rds::RealEnvironment>("../data/layout/real_factory/s_tech", 6, 1.0, 200, ax1, true);
    // // rds::EnvironmentInterfaceHandle env = std::make_shared<rds::SingleEnvironment>("../data/layout/single/60x30", 60, 1.0, 200, ax1, true);
    // ax1->axes_aspect_ratio(env->map_width()/env->map_length());
    // ax1->position({0.0f, 0.0f, 1.0f, 1.0f});
    // fig1->size(int(env->map_length()) * 8, int(env->map_width()) * 8);
    // // fig1->size(int(env->map_length()) * 25, int(env->map_width()) * 25);
    // ax1->xticks({});
    // ax1->yticks({});
    // env->visualize();
    // rds::TaskGeneratorHandle task_gen = std::make_shared<rds::TaskGenerator>(5, 5, 2, 10.0f, 200.0f, env->graph_handle());
    // rds::AllocationInterfaceHandle allocation = std::make_shared<rds::AstarAllocation>(env->swarm(), env->graph_handle(), 
    //                                                                                     task_gen, 1, 1, env->map_center(), 
    //                                                                                     env->map_length(), env->map_width(),
    //                                                                                     env->single_line_width(), 
    //                                                                                     env->double_line_width());
    // rds::PlannerInterfaceHandle planner = std::make_shared<rds::AStarPlannerWithWaiting>(env->swarm(), env->graph_handle(), 
    //                                                                                     1, 1, env->map_center(), env->map_length(), 
    //                                                                                     env->map_width(), env->single_line_width(), 
    //                                                                                     env->double_line_width());
    // // Define the codec and create VideoWriter object.
    // int fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
    // std::string output_file = "../data/visualize.avi";
    // cv::VideoWriter video(output_file, fourcc, 12, cv::Size(int(env->map_length()) * 8, int(env->map_width()) * 8));
    // // allocation->allocation();
    // // planner->planning();
    // // for (int i = 0; i < env->swarm().size(); i++)
    // // {
    // //     std::cout << "Robot " << env->swarm()[i]->id() << ": " << env->swarm()[i]->route_ids().size() << " " << env->swarm()[i]->route_positions().size() << std::endl;
    // //     // matplot::vector_2d route = rds::point2d_to_plot_2d(env->swarm()[i]->route_positions());
    // //     // matplot::line_handle route_visual = ax1->plot(route[0], route[1]);
    // //     // route_visual->color(rds::RED);
    // //     // route_visual->line_width(2);
    // // }
    // // std::cout << task_gen->task_queue().size() << std::endl;
    // // while (true)
    // // std::cout << env->swarm()[0]->pose() << std::endl;
    // for(int i = 0; i < 5000; i++)
    // {
    //     allocation->allocation();
    //     planner->planning();
    //     env->controller(3);
    //     env->visualize();
    //     matplot::save("../include/evaluation/factory_map.png");
    //     std::this_thread::sleep_for(std::chrono::milliseconds(100));
    //     cv::Mat image = cv::imread("../include/evaluation/factory_map.png");
    //     // cv::imshow("Visualize", image);
    //     video.write(image);
    //     // cv::waitKey(0);
    //     // if (cv::waitKey(1) && 0xFF == 'q') break;
    // }
    
    // // Close the video file.
    // video.release();
    return 0;
}