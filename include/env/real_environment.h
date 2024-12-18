#pragma once
#ifndef REAL_ENVIRONMENT_H_
#define REAL_ENVIRONMENT_H_

#include "env/environment_interface.h"
namespace rds
{
    class RealEnvironment : public EnvironmentInterface
    {
    private:
        float dt_;
        ReconstructInterfaceHandle factory_;
        int num_robots_;
        float d_safe_junction_, d_safe_closest_;
        int num_zones_in_col_, num_zones_in_row_;
        std::vector<GraphZoneHandle> graph_zones_;
        SwarmHandle swarm_;
        GraphHandle graph_;

    public:
        RealEnvironment(std::string data_path, int num_robots, float robot_max_speed, float robot_max_payload,
                        matplot::axes_handle visual, bool show_edge = false);
        virtual void reset(){};
        virtual void controller(float wait_time);
        bool task_assign_control_state(RobotHandle robot, float wait_time);
        void traffic_assign_control_state(RobotHandle robot);
        void set_control_state_to_pass_same_point(RobotHandle robot);
        void calculate_same_and_between(RobotHandle robot);
        std::vector<int> filter_same_vertex_list(RobotHandle robot, std::vector<int> same_vertex_list);
        Point2D calculate_wait_point(RobotHandle robot, Point2D target_point, float distance);
        bool line_is_safe(RobotHandle robot, Point2D start_point, Point2D target_point);
        bool vertex_is_safe(RobotHandle robot, Point2D next_point);
        bool compare_by_num_between(int id1, int id2);
        virtual SwarmHandle swarm();
        virtual GraphHandle graph_handle();
        virtual Graph graph();
        virtual void visualize();
        virtual Point2D map_center();
        virtual float map_length();
        virtual float map_width();
        virtual float single_line_width();
        virtual float double_line_width();
        virtual float dt() {return dt_;};
        virtual void add_all_robot_graph(){};
    };
}
#endif