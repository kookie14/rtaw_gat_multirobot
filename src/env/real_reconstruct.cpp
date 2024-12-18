#include "env/real_reconstruct.h"

namespace rds
{
    RealReconstruction::RealReconstruction(std::string data_path)
    {
        data_path_ = data_path;
        vel_list_ = {0.6, 0.7, 0.8, 0.9, 1.0};
        read_map_data();
        build_connected_graph();
    }

    void RealReconstruction::set_data_path(std::string data_path)
    {
        data_path_ = data_path;
        read_map_data();
        build_connected_graph();
    }
    Graph RealReconstruction::graph() {return graph_;}
    Vertices RealReconstruction::waiting_vertices() {return wa_vertices_;}
    Point2D RealReconstruction::map_center() {return map_center_;}
    float RealReconstruction::map_width() {return map_width_;}
    float RealReconstruction::map_length() {return map_length_;}
    float RealReconstruction::single_line_width() {return single_line_width_;}
    float RealReconstruction::double_line_width() {return double_line_width_;}

    void RealReconstruction::read_map_data()
    {
        std::vector<std::vector<float>> map_data = load_txt(data_path_ + "/map.txt");
        map_center_ = Point2D(map_data[0][0], map_data[1][0]);
        map_length_ = map_data[2][0];
        map_width_ = map_data[3][0];
        single_line_width_ = map_data[4][0];
        double_line_width_ = map_data[5][0];
        map_coords_ = vector_to_plot_2d(calculate_rectangle_coordinate(map_center_.x(), map_center_.y(), 0.0, map_length_ + 0.5, map_width_ + 0.5));
        std::vector<std::vector<float>> line_data = load_txt(data_path_ + "/line_data.txt");
        for (int i = 0; i < line_data.size(); i++)
        {
            add_line(int(line_data[i][0]), Point2D(line_data[i][1], line_data[i][2]), line_data[i][3]);
        }
        std::vector<std::vector<float>> point_data = load_txt(data_path_ + "/point_data.txt");
        for (int i = 0; i < point_data.size(); i++)
        {
            add_zone_point(int(point_data[i][0]), int(point_data[i][1]), Point2D(point_data[i][2], point_data[i][3]), point_data[i][4], point_data[i][5]);
            z_centers_.push_back({z_points_[i].x(), z_points_[i].y()});
        }
    }

    void RealReconstruction::build_connected_graph()
    {
        for (int i = 0; i < s_h_lines_.size(); i++)
            s_h_vertices_.push_back(Vertices());
        for (int i = 0; i < s_v_lines_.size(); i++)
            s_v_vertices_.push_back(Vertices());

        build_graph_vertices();
        build_graph_edges();
        graph_.set_fixed_vertices_and_edges(graph_.vertices().size(), graph_.edges().size());
    }

    void RealReconstruction::build_graph_vertices()
    {
        build_graph_vertices_from_zone();
        build_graph_vertices_from_single_horizontal_line();
        build_graph_vertices_from_single_vertical_line();
        build_graph_vertices_from_intersection_line();
    }

    void RealReconstruction::build_graph_edges()
    {
        build_graph_edges_from_zone_and_lines();
        build_graph_edges_from_single_horizontal_line();
        build_graph_edges_from_single_vertical_line();
    }

    void RealReconstruction::build_graph_vertices_from_zone()
    {
        for (Point point : w_points_)
        {
            graph_.add_vertex(WORKING_VERTEX, point.center());
            z_vertices_.push_back(graph_.vertex(-1));
            graph_points_.push_back({point.x(), point.y()});
        }
        for (Point point : s_points_)
        {
            graph_.add_vertex(STORAGE_VERTEX, point.center());
            z_vertices_.push_back(graph_.vertex(-1));
            graph_points_.push_back({point.x(), point.y()});
        }
        for (Point point : c_points_)
        {
            graph_.add_vertex(CHARGING_VERTEX, point.center());
            z_vertices_.push_back(graph_.vertex(-1));
            graph_points_.push_back({point.x(), point.y()});
        }
        for (Point point : wa_points_)
        {
            graph_.add_vertex(WAITING_VERTEX, point.center(), 0, 0);
            z_vertices_.push_back(graph_.vertex(-1));
            graph_points_.push_back({point.x(), point.y()});
            wa_vertices_.push_back(graph_.vertex(-1));
        }
    }

    void RealReconstruction::build_graph_vertices_from_single_horizontal_line()
    {
        for (int i = 0; i < s_h_lines_.size(); i++)
        {
            Line line = s_h_lines_[i];
            float point_y1 = round_num(line.y_center() + single_line_width_ * 0.5 + w_points_[0].length() * 0.5);
            float point_y2 = round_num(line.y_center() - single_line_width_ * 0.5 - w_points_[0].length() * 0.5);
            std::vector<int> y1_indices, y2_indices;
            for (int j = 0; j < z_vertices_.size(); ++j)
            {
                if (fabs(z_vertices_[j].y() - point_y1) < SAME_DIST)
                    y1_indices.push_back(j);
                if (fabs(z_vertices_[j].y() - point_y2) < SAME_DIST)
                    y2_indices.push_back(j);
            }

            for (int idx : y1_indices)
            {
                if (line.x_center() - line.length() / 2 <= z_vertices_[idx].x() && z_vertices_[idx].x() <= line.x_center() + line.length() / 2)
                {
                    if (!vector_exists(graph_points_, {z_vertices_[idx].x(), line.y_center()}))
                    {
                        graph_points_.push_back({z_vertices_[idx].x(), line.y_center()});
                        graph_.add_vertex(LINE_VERTEX, Point2D(z_vertices_[idx].x(), line.y_center()), UNDIRECTED, UNDIRECTED, SINGLE_HORIZONTAL);
                        s_h_vertices_[i].push_back(graph_.vertex(-1));
                    }
                }
            }
            for (int idx : y2_indices)
            {
                if (line.x_center() - line.length() / 2 <= z_vertices_[idx].x() && z_vertices_[idx].x() <= line.x_center() + line.length() / 2)
                {
                    if (!vector_exists(graph_points_, {z_vertices_[idx].x(), line.y_center()}))
                    {
                        graph_points_.push_back({z_vertices_[idx].x(), line.y_center()});
                        graph_.add_vertex(LINE_VERTEX, Point2D(z_vertices_[idx].x(), line.y_center()), UNDIRECTED, UNDIRECTED, SINGLE_HORIZONTAL);
                        s_h_vertices_[i].push_back(graph_.vertex(-1));
                    }
                }
            }
        }
    }

    void RealReconstruction::build_graph_vertices_from_single_vertical_line()
    {
        for (int i = 0; i < s_v_lines_.size(); i++)
        {
            Line line = s_v_lines_[i];
            float point_x1 = round_num(line.x_center() + single_line_width_ * 0.5 + w_points_[0].length() * 0.5);
            float point_x2 = round_num(line.x_center() - single_line_width_ * 0.5 - w_points_[0].length() * 0.5);
            std::vector<int> x1_indices, x2_indices;
            for (int j = 0; j < z_vertices_.size(); ++j)
            {
                if (fabs(z_vertices_[j].x() - point_x1) < SAME_DIST)
                    x1_indices.push_back(j);
                if (fabs(z_vertices_[j].x() - point_x2) < SAME_DIST)
                    x2_indices.push_back(j);
            }

            for (int idx : x1_indices)
            {
                if (line.y_center() - line.length() / 2 <= z_vertices_[idx].y() && z_vertices_[idx].y() <= line.y_center() + line.length() / 2)
                {
                    if (!vector_exists(graph_points_, {line.x_center(), z_vertices_[idx].y()}))
                    {
                        graph_points_.push_back({line.x_center(), z_vertices_[idx].y()});
                        graph_.add_vertex(LINE_VERTEX, Point2D(line.x_center(), z_vertices_[idx].y()), UNDIRECTED, UNDIRECTED, SINGLE_VERTICAL);
                        s_v_vertices_[i].push_back(graph_.vertex(-1));
                    }
                }
            }
            for (int idx : x2_indices)
            {
                if (line.y_center() - line.length() / 2 <= z_vertices_[idx].y() && z_vertices_[idx].y() <= line.y_center() + line.length() / 2)
                {
                    if (!vector_exists(graph_points_, {line.x_center(), z_vertices_[idx].y()}))
                    {
                        graph_points_.push_back({line.x_center(), z_vertices_[idx].y()});
                        graph_.add_vertex(LINE_VERTEX, Point2D(line.x_center(), z_vertices_[idx].y()), UNDIRECTED, UNDIRECTED, SINGLE_VERTICAL);
                        s_v_vertices_[i].push_back(graph_.vertex(-1));
                    }
                }
            }
        }
    }

    void RealReconstruction::build_graph_vertices_from_intersection_line()
    {
        for (int i = 0; i < s_v_lines_.size(); i++)
        {
            build_graph_vertices_from_s_v_and_s_h(s_v_lines_[i], i);
        }
    }

    void RealReconstruction::build_graph_vertices_from_s_v_and_s_h(Line v_l, int line_idx)
    {
        for (int i = 0; i < s_h_lines_.size(); i++)
        {
            Line s_l = s_h_lines_[i];
            if (v_l.y_center() - v_l.length() / 2 - single_line_width_ <= s_l.y_center() && s_l.y_center() <= v_l.y_center() + v_l.length() / 2 + single_line_width_)
            {
                if (s_l.x_center() - s_l.length() / 2 - single_line_width_ <= v_l.x_center() && v_l.x_center() <= s_l.x_center() + s_l.length() / 2 + single_line_width_)
                {
                    if (!vector_exists(graph_points_, {v_l.x_center(), s_l.y_center()}))
                    {
                        graph_points_.push_back({v_l.x_center(), s_l.y_center()});
                        graph_.add_vertex(LINE_VERTEX, Point2D(v_l.x_center(), s_l.y_center()), UNDIRECTED, UNDIRECTED, MIX_LINE);
                        s_h_vertices_[i].push_back(Vertex(graph_.vertex(-1).id(), LINE_VERTEX, graph_.vertex(-1).position(),
                                                          UNDIRECTED, UNDIRECTED, SINGLE_HORIZONTAL));
                        s_v_vertices_[line_idx].push_back(Vertex(graph_.vertex(-1).id(), LINE_VERTEX, graph_.vertex(-1).position(),
                                                                 UNDIRECTED, UNDIRECTED, SINGLE_VERTICAL));
                    }
                }
            }
        }
    }

    void RealReconstruction::build_graph_edges_from_zone_and_lines()
    {
        build_graph_edges_from_zone_and_line(s_h_vertices_);
        build_graph_edges_from_zone_and_line(s_v_vertices_);
    }

    void RealReconstruction::build_graph_edges_from_zone_and_line(ListVertices vertices)
    {
        int idx = torch::randint(0, vel_list_.size() - 1, 1).item().toInt();
        for (int i = 0; i < vertices.size(); i++)
        {
            for (int j = 0; j < vertices[i].size(); j++)
            {
                for (int k = 0; k < z_vertices_.size(); k++)
                {
                    if (is_edge_zone_line(vertices[i][j], z_vertices_[k]) == true)
                    {
                        if (!number_exists(graph_.neighbors(vertices[i][j].id()), z_vertices_[k].id()))
                            graph_.add_edge(vertices[i][j].id(), z_vertices_[k].id(), vel_list_[idx]);
                        if (!number_exists(graph_.neighbors(z_vertices_[k].id()), vertices[i][j].id()))
                            graph_.add_edge(z_vertices_[k].id(), vertices[i][j].id(), vel_list_[idx]);
                    }
                }
            }
        }
    }

    bool RealReconstruction::is_edge_zone_line(Vertex v1, Vertex v2)
    {
        bool cond1 = v1.x() == v2.x();
        bool cond2 = v1.y() == v2.y();
        float max_length = w_points_[0].length();
        if (s_points_.size() != 0) max_length = std::max<float>(max_length, s_points_[0].length());
        if (wa_points_.size() != 0) max_length = std::max<float>(max_length, wa_points_[0].length());
        if (c_points_.size() != 0) max_length = std::max<float>(max_length, c_points_[0].length());
        max_length = max_length * 0.5 + double_line_width_ * 0.25 + 0.1;
        if ((cond1 || cond2) && euclidean_distance(v1.position(), v2.position()) <= max_length)
            return true;
        return false;
    }

    void RealReconstruction::build_graph_edges_from_single_horizontal_line()
    {
        for (Vertices list : s_h_vertices_)
        {
            Vertices v_list = list;
            int idx = torch::randint(0, vel_list_.size() - 1, 1).item().toInt();
            std::sort(v_list.begin(), v_list.end(), compare_vertex_by_x);
            for (int i = 0; i < v_list.size() - 1; i++)
            {
                if (!number_exists(graph_.neighbors(v_list[i].id()), v_list[i + 1].id()))
                    graph_.add_edge(v_list[i].id(), v_list[i + 1].id(), vel_list_[idx]);

                if (!number_exists(graph_.neighbors(v_list[i + 1].id()), v_list[i].id()))
                    graph_.add_edge(v_list[i + 1].id(), v_list[i].id(), vel_list_[idx]);
            }
        }
    }

    void RealReconstruction::build_graph_edges_from_single_vertical_line()
    {
        for (Vertices list : s_v_vertices_)
        {
            Vertices v_list = list;
            int idx = torch::randint(0, vel_list_.size() - 1, 1).item().toInt();
            std::sort(v_list.begin(), v_list.end(), compare_vertex_by_y);
            for (int i = 0; i < v_list.size() - 1; i++)
            {
                if (!number_exists(graph_.neighbors(v_list[i].id()), v_list[i + 1].id()))
                    graph_.add_edge(v_list[i].id(), v_list[i + 1].id(), vel_list_[idx]);

                if (!number_exists(graph_.neighbors(v_list[i + 1].id()), v_list[i].id()))
                    graph_.add_edge(v_list[i + 1].id(), v_list[i].id(), vel_list_[idx]);
            }
        }
    }

    void RealReconstruction::add_line(int type, Point2D center, float length)
    {
        if (type == SINGLE_HORIZONTAL)
        {
            s_h_lines_.push_back(Line(type, center, length, single_line_width_));
            lines_.push_back(Line(type, center, length, single_line_width_));
        }
        else if (type == SINGLE_VERTICAL)
        {
            s_v_lines_.push_back(Line(type, center, length, single_line_width_));
            lines_.push_back(Line(type, center, length, single_line_width_));
        }
    }

    void RealReconstruction::add_zone_point(int point_type, int line_type, Point2D center, float length, float width)
    {
        z_points_.push_back(Point(point_type, line_type, center, length, width));
        if (point_type == WORKING_VERTEX)
            w_points_.push_back(Point(point_type, line_type, center, length, width));
        else if (point_type == STORAGE_VERTEX)
            s_points_.push_back(Point(point_type, line_type, center, length, width));
        else if (point_type == WAITING_VERTEX)
            wa_points_.push_back(Point(point_type, line_type, center, length, width));
        else if (point_type == CHARGING_VERTEX)
            c_points_.push_back(Point(point_type, line_type, center, length, width));
    }

    void RealReconstruction::visualize(matplot::axes_handle visual, bool show_edge)
    {
        visual->hold(true);
        auto min_x = std::min_element(map_coords_[0].begin(), map_coords_[0].end());
        auto max_x = std::max_element(map_coords_[0].begin(), map_coords_[0].end());
        auto min_y = std::min_element(map_coords_[1].begin(), map_coords_[1].end());
        auto max_y = std::max_element(map_coords_[1].begin(), map_coords_[1].end());
        int length_addition = 1.0;
        visual->axis({*min_x - length_addition, *max_x + length_addition, *min_y - length_addition, *max_y + length_addition});
        auto map = visual->fill(map_coords_[0], map_coords_[1]);
        map->color(WHITE);
        // map->line_width(1.5);
        // for (int i = 0; i < lines_.size(); i++)
        // {
        //     V2d vec = vector_to_plot_2d(lines_[i].coords());
        //     auto line = visual->fill(vec[0], vec[1]);
        //     line->color(WHITE);
        // }
        // for (int i = 0; i < z_points_.size(); i++)
        // {
        //     V2d vec = vector_to_plot_2d(z_points_[i].coords());
        //     auto point = visual->fill(vec[0], vec[1]);
        //     point->color(WHITE);
        // }
        V1d x_w, y_w, x_wa, y_wa, x_s, y_s, x_c, y_c, x_l, y_l, x_r, y_r;
        for (Vertex v : graph_.vertices())
        {
            switch (v.type())
            {
            case WORKING_VERTEX:
                x_w.push_back(v.x());
                y_w.push_back(v.y());
                break;
            case STORAGE_VERTEX:
                x_s.push_back(v.x());
                y_s.push_back(v.y());
                break;
            case WAITING_VERTEX:
                x_wa.push_back(v.x());
                y_wa.push_back(v.y());
                break;
            case CHARGING_VERTEX:
                x_c.push_back(v.x());
                y_c.push_back(v.y());
                break;
            // case LINE_VERTEX:
            //     x_l.push_back(v.x());
            //     y_l.push_back(v.y());
            //     break;
            default:
                break;
            }
        }
        auto p_w = visual->plot(x_w, y_w, ".");
        p_w->color(GREEN);
        auto p_wa = visual->plot(x_wa, y_wa, ".");
        p_wa->color(RED);
        if (x_s.size() != 0)
        {
            auto p_s = visual->plot(x_s, y_s, ".");
            p_s->color(BLUE);
        }
        auto p_c = visual->plot(x_c, y_c, ".");
        p_c->color(ORANGE);
        // auto p_l = visual->plot(x_l, y_l, ".");
        // p_l->color(PURPLE);
        if (show_edge == true)
        {
            for (Edge edge : graph_.edges())
            {
                auto arrow = visual->arrow(graph_.vertex(edge.start_id()).x(), graph_.vertex(edge.start_id()).y(),
                                            graph_.vertex(edge.end_id()).x(), graph_.vertex(edge.end_id()).y());
                arrow->color(GREY);
                arrow->line_width(0.5);
            }
        }
        
    }
}