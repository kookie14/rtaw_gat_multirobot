#pragma once
#ifndef RECONSTRUCT_INTERFACE_H_
#define RECONSTRUCT_INTERFACE_H_

#include "utils/type_define.h"

namespace rds
{
    class ReconstructInterface
    {
    public:
        virtual void read_map_data() = 0;
        virtual void build_connected_graph() = 0;
        virtual void set_data_path(std::string data_path = "") = 0;
        virtual Graph graph() = 0;
        virtual Vertices waiting_vertices() = 0;
        virtual Point2D map_center() = 0;
        virtual float map_width() = 0;
        virtual float map_length() = 0;
        virtual float single_line_width() = 0;
        virtual float double_line_width() = 0;
        virtual void visualize(matplot::axes_handle visual, bool show_edge = false) = 0;
    };

    typedef std::shared_ptr<ReconstructInterface> ReconstructInterfaceHandle;
}
#endif