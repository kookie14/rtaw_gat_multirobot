#pragma once
#ifndef REAL_RECONSTRUCT_H_
#define REAL_RECONSTRUCT_H_

#include "env/reconstruct_interface.h"
namespace rds
{
    class RealReconstruction: public ReconstructInterface
    {
    private:
        std::string data_path_;
        float single_line_width_;
        float double_line_width_;
        Graph graph_;
        // Map data
        Point2D map_center_;
        float map_length_;
        float map_width_;
        V2d map_coords_;
        // all lines, single horizontal, single vertical
        Lines lines_, s_h_lines_, s_v_lines_;
        // zone points, working points, storage points, waiting points, charging points
        Points z_points_, w_points_, s_points_, wa_points_, c_points_;
        std::vector<std::vector<float>> z_centers_;
        // zone, waiting, single horizontal line, single vertical line vertices
        Vertices z_vertices_, wa_vertices_;
        ListVertices s_h_vertices_, s_v_vertices_;
        // Graph points
        std::vector<std::vector<float>> graph_points_;
        std::vector<float> vel_list_;

    public:
        RealReconstruction(std::string data_path = "");
        virtual void set_data_path(std::string data_path = "");
        virtual Graph graph();
        virtual Vertices waiting_vertices();
        virtual Point2D map_center();
        virtual float map_width();
        virtual float map_length();
        virtual float single_line_width();
        virtual float double_line_width();
        virtual void read_map_data();
        void add_line(int type, Point2D center, float length);
        void add_zone_point(int point_type, int line_type, Point2D center, float length, float width);
        virtual void build_connected_graph();
        void build_graph_vertices();
        void build_graph_edges();
        void build_graph_vertices_from_single_horizontal_line();
        void build_graph_vertices_from_single_vertical_line();
        void build_graph_vertices_from_intersection_line();
        void build_graph_vertices_from_s_v_and_s_h(Line v_l, int line_idx); // single vertical line intersect with single horizontal line
        void build_graph_edges_from_zone_and_line(ListVertices vertices);
        void build_graph_vertices_from_zone();
        void build_graph_edges_from_zone_and_lines();
        bool is_edge_zone_line(Vertex v1, Vertex v2);
        void build_graph_edges_from_single_horizontal_line();
        void build_graph_edges_from_single_vertical_line();
        virtual void visualize(matplot::axes_handle visual, bool show_edge = false);
    };
}
#endif