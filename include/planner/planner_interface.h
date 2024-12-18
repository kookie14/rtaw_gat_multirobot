#pragma once
#ifndef PLANNING_INTERFACE_H_
#define PLANNING_INTERFACE_H_

#include "robot/robot.h"

namespace rds
{
    class PlannerInterface
    {
    public:
        virtual void planning() = 0;
    };
    typedef std::shared_ptr<PlannerInterface> PlannerInterfaceHandle;
}

#endif