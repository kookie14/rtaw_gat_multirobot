#pragma once
#ifndef TYPE_AND_STATE_H_
#define TYPE_AND_STATE_H_
#include <vector>
#include <string>
namespace rds
{
// Line type
const int NONE_LINE = -1;
const int DOUBLE_HORIZONTAL = 0;
const int SINGLE_HORIZONTAL = 1;
const int DOUBLE_VERTICAL = 2;
const int SINGLE_VERTICAL = 3;
const int MIX_LINE = 4;
// Zone type
const int NONE_ZONE = -1;
const int WORKING_ZONE = 0;
const int STORAGE_ZONE = 1;
const int WAITING_ZONE = 2;
const int CHARGING_ZONE = 3;
const int LINE_ZONE = 4;
// Graph vertex type
const int UNKNOWN_VERTEX = -1;
const int LINE_VERTEX = 0;
const int WORKING_VERTEX = 1;
const int STORAGE_VERTEX = 2;
const int WAITING_VERTEX = 3;
const int CHARGING_VERTEX = 4;
const int BOUNDARY_VERTEX = 5;
const int ROBOT_VERTEX = 6;
// Allocation state
// std::vector<std::string> state_color = {"lime", "olive", "gold", "chocolate", "navy", "cyan", "orange", "red", "blue", "brown", "black"};
const std::array<float, 4> RED = {0.0f, 1.0f, 0.0f, 0.0f};
const std::array<float, 4> BLUE = {0.0f, 0.0f, 0.0f, 1.0f};
const std::array<float, 4> GREEN = {0.0f, 0.0f, 1.0f, 0.0f};
const std::array<float, 4> YELLOW = {0.0f, 1.0f, 1.0f, 0.0f};
const std::array<float, 4> OLIVE = {0.0f, 0.5f, 0.5f, 0.0f};
const std::array<float, 4> GOLD = {0.0f, 1.0f, 0.843f, 0.0f};
const std::array<float, 4> CHOCOLATE = {0.0f, 0.8235f, 0.4118f, 0.1176f};
const std::array<float, 4> NAVY = {0.0f, 0.0f, 0.0f, 0.5f};
const std::array<float, 4> CYAN = {0.0f, 0.0f, 1.0f, 1.0f};
const std::array<float, 4> ORANGE = {0.0f, 1.0f, 0.6471f, 0.0f};
const std::array<float, 4> BROWN = {0.0f, 0.6471f, 0.1647f, 0.1647f};
const std::array<float, 4> BLACK = {0.0f, 0.0f, 0.0f, 0.0f};
const std::array<float, 4> WHITE = {0.0f, 1.0f, 1.0f, 1.0f};
const std::array<float, 4> PURPLE = {0.0f, 0.5f, 0.0f, 0.5f};
const std::array<float, 4> GREY = {0.0f, 0.5f, 0.5f, 0.5f};
const int FREE = 0;
const int ON_WAY_TO_START = 1;
const int ON_WAY_TO_TARGET = 2;
const int ON_WAY_TO_WAITING = 3;
const int ON_WAY_TO_CHARGING = 4;
const int PICKING_UP = 5;
const int BUSY = 6;
const int LOW_BATTERY = 7;
const int CHARGING = 8;
const int AVOIDANCE = 9;
const int DIE = 10;
// Control state
const int TO_NEXT_POINT = 0;
const int TO_WAIT_POINT = 1;
const int WAIT = 2;
const int WAIT_BY_GOAL_OCCUPIED = 3;
// Direct of a edge
const int UNDIRECTED = 0;
const int POSITIVE_DIRECTED = 1;
const int NEGATIVE_DIRECTED = 2;
// Route type
const int NONE_ROUTE = -1;
const int TO_START = 0;
const int TO_TARGET = 1;
const int TO_WAITING = 2;
const int TO_CHARGING = 3;
// Robot initialization mode
const int WAITING_INITIALIZATION = 0;
const int RANDOM_INITIALIZATION = 1;
const int TEST_INITIALIZATION = 2;
// Task generate mode
const int DETERMINISTIC_MODE = 0;
const int RANDOM_MODE = 1;
// Learning mode
const int TRAINING = 0;
const int TESTING = 1;
}
#endif