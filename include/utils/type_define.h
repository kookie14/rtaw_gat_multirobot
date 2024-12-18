#pragma once
#ifndef TYPE_DEFINE_H_
#define TYPE_DEFINE_H_
#include "utils/map_and_graph.h"

namespace rds
{
typedef std::vector<Line> Lines;
typedef std::vector<Point> Points;
typedef matplot::vector_1d V1d;
typedef matplot::vector_2d V2d;
typedef std::vector<Vertex> Vertices;
typedef std::vector<Vertices> ListVertices;
typedef std::shared_ptr<Pose2D> Pose2DHandle;
typedef std::shared_ptr<Point2D> Point2DHandle;
typedef std::shared_ptr<Velocity> VelocityHandle;
typedef std::shared_ptr<Task> TaskHandle;
typedef std::shared_ptr<Route> RouteHandle;
typedef std::shared_ptr<Graph> GraphHandle;
typedef std::shared_ptr<GraphZone> GraphZoneHandle;
typedef std::vector<TaskHandle> TaskHandleList;
}

#endif