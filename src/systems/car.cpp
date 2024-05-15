/**
 * @file car.cpp
 *
 * @copyright Software License Agreement (BSD License)
 * Original work Copyright (c) 2014, Rutgers the State University of New Jersey, New Brunswick
 * Modified work Copyright 2017 Oleg Y. Sinyavskiy
 * All Rights Reserved.
 * For a full description see the file named LICENSE.
 *
 * Original authors: Zakary Littlefield, Kostas Bekris
 * Modifications by: Oleg Y. Sinyavskiy
 *
 */

#include "systems/car.hpp"
#include "utilities/random.hpp"

#define WIDTH 2.0
#define LENGTH 1.0
#define STATE_X 0
#define STATE_Y 1
#define STATE_THETA 2
#define MIN_X -25
#define MAX_X 25
#define MIN_Y -35
#define MAX_Y 35

#define _USE_MATH_DEFINES

#include <cmath>


bool car_t::propagate(
    const double* start_state, unsigned int state_dimension,
    const double* control, unsigned int control_dimension,
    int num_steps, double* result_state, double integration_step)
{
	temp_state[0] = start_state[0]; temp_state[1] = start_state[1];temp_state[2] = start_state[2];

	bool validity = true;
	for(int i=0;i<num_steps;i++)
	{
        update_derivative(control);
        temp_state[0] += integration_step*deriv[0];
        temp_state[1] += integration_step*deriv[1];
        temp_state[2] += integration_step*deriv[2];
		enforce_bounds();
		validity = validity && valid_state();
	}
	result_state[0] = temp_state[0];
	result_state[1] = temp_state[1];
	result_state[2] = temp_state[2];
	return validity;
}

void car_t::update_derivative(const double* control)
{
    // angle: clockwise
    deriv[0] = cos(temp_state[2]) * control[0];
    deriv[1] = -sin(temp_state[2]) * control[0];
    deriv[2] = control[1];
}


void car_t::enforce_bounds()
{
    /*
	if(temp_state[0]<-10)
		temp_state[0]=-10;
	else if(temp_state[0]>10)
		temp_state[0]=10;

	if(temp_state[1]<-10)
		temp_state[1]=-10;
	else if(temp_state[1]>10)
		temp_state[1]=10;
    */
	if(temp_state[2]<-M_PI)
		temp_state[2]+=2*M_PI;
	else if(temp_state[2]>M_PI)
		temp_state[2]-=2*M_PI;
}

bool car_t::valid_state()
{
    if (temp_state[0] < MIN_X || temp_state[0] > MAX_X || temp_state[1] < MIN_Y || temp_state[1] > MAX_Y)
    {
        return false;
    }
    return true;
}

std::tuple<double, double> car_t::visualize_point(const double* state, unsigned int state_dimension) const
{
	double x = (state[0]+10)/(20);
	double y = (state[1]+10)/(20);
	return std::make_tuple(x, y);
}

std::vector<std::pair<double, double> > car_t::get_state_bounds() const {
	return {
        {MIN_X,MAX_X},
        {MIN_Y,MAX_Y},
        {-M_PI,M_PI},
	};
}


std::vector<std::pair<double, double> > car_t::get_control_bounds() const {
    return {
            {0, 2},
            {-.5,.5},
    };
}

std::vector<bool> car_t::is_circular_topology() const {
	return {
			false,
			false,
			true
	};
}
