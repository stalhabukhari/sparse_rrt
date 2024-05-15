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

#include "systems/car_obs.hpp"
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


bool car_obs_t::propagate(
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
    //std::cout << "after propagation" << std::endl;
	return validity;
}

void car_obs_t::update_derivative(const double* control)
{
    deriv[0] = cos(temp_state[2]) * control[0];
    deriv[1] = -sin(temp_state[2]) * control[0];
    deriv[2] = control[1];
}


void car_obs_t::enforce_bounds()
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

bool car_obs_t::overlap(std::vector<std::vector<double>>& b1corner, std::vector<std::vector<double>>& b1axis,
                        std::vector<double>& b1orign, std::vector<double>& b1ds,
                        std::vector<std::vector<double>>& b2corner, std::vector<std::vector<double>>& b2axis,
                        std::vector<double>& b2orign, std::vector<double>& b2ds)
{
    for (unsigned a = 0; a < 2; a++)
    {
        double t = b1corner[0][0]*b2axis[a][0] + b1corner[0][1]*b2axis[a][1];
        double tMin = t;
        double tMax = t;
        for (unsigned c = 1; c < 4; c++)
        {
            t = b1corner[c][0]*b2axis[a][0]+b1corner[c][1]*b2axis[a][1];
            if (t < tMin)
            {
                tMin = t;
            }
            else if (t > tMax)
            {
                tMax = t;
            }
        }
        if ((tMin > (b2ds[a] + b2orign[a])) || (tMax < b2orign[a]))
        {
            return false;
        }
    }
    return true;

}

bool car_obs_t::valid_state()
{
    if (temp_state[0] < MIN_X || temp_state[0] > MAX_X || temp_state[1] < MIN_Y || temp_state[1] > MAX_Y)
    {
        return false;
    }
    //std::cout << "inside  valid_state" << std::endl;

    std::vector<std::vector<double>> robot_corner(4, std::vector<double> (2, 0));
    std::vector<std::vector<double>> robot_axis(2, std::vector<double> (2,0));
    std::vector<double> robot_ori(2, 0);
    std::vector<double> length(2, 0);
    std::vector<double> X1(2,0);
    std::vector<double> Y1(2,0);

    X1[0]=cos(temp_state[STATE_THETA])*(WIDTH/2.0);
    X1[1]=-sin(temp_state[STATE_THETA])*(WIDTH/2.0);
    Y1[0]=sin(temp_state[STATE_THETA])*(LENGTH/2.0);
    Y1[1]=cos(temp_state[STATE_THETA])*(LENGTH/2.0);

    for (unsigned j = 0; j < 2; j++)
    {
        // order: (left-bottom, right-bottom, right-upper, left-upper)
        robot_corner[0][j]=temp_state[j]-X1[j]-Y1[j];
        robot_corner[1][j]=temp_state[j]+X1[j]-Y1[j];
        robot_corner[2][j]=temp_state[j]+X1[j]+Y1[j];
        robot_corner[3][j]=temp_state[j]-X1[j]+Y1[j];
        //axis: horizontal and vertical
        robot_axis[0][j] = robot_corner[1][j] - robot_corner[0][j];
        robot_axis[1][j] = robot_corner[3][j] - robot_corner[0][j];
    }

    length[0]=sqrt(robot_axis[0][0]*robot_axis[0][0]+robot_axis[0][1]*robot_axis[0][1]);
    length[1]=sqrt(robot_axis[1][0]*robot_axis[1][0]+robot_axis[1][1]*robot_axis[1][1]);

    for (unsigned i=0; i<2; i++)
    {
        for (unsigned j=0; j<2; j++)
        {
            robot_axis[i][j]=robot_axis[i][j]/length[i];
        }
    }
    // obtain the projection of the left-bottom corner to the axis, to obtain the minimal projection length
    robot_ori[0]=robot_corner[0][0]*robot_axis[0][0]+ robot_corner[0][1]*robot_axis[0][1];
    robot_ori[1]=robot_corner[0][0]*robot_axis[1][0]+ robot_corner[0][1]*robot_axis[1][1];

    static std::vector<double> car_size{WIDTH, LENGTH};
    static std::vector<double> obs_size{this->obs_width, this->obs_width};

    for (unsigned i=0; i<obs_list.size(); i++)
    {
        bool collision = true;
        // do checking in both direction (b1 -> b2, b2 -> b1). It is only collision if both direcions are collision
        collision = overlap(robot_corner,robot_axis,robot_ori,car_size,\
                            obs_list[i],obs_axis[i],obs_ori[i],obs_size);
        collision = collision&overlap(obs_list[i],obs_axis[i],obs_ori[i],obs_size,\
                                      robot_corner,robot_axis,robot_ori,car_size);
        if (collision)
        {
            return false;  // invalid state
        }
    }
    //std::cout << "after valid" << std::endl;

    return true;



}

std::tuple<double, double> car_obs_t::visualize_point(const double* state, unsigned int state_dimension) const
{
	double x = (state[0]+10)/(20);
	double y = (state[1]+10)/(20);
	return std::make_tuple(x, y);
}

std::vector<std::pair<double, double> > car_obs_t::get_state_bounds() const {
	return {
        {MIN_X,MAX_X},
        {MIN_Y,MAX_Y},
        {-M_PI,M_PI},
	};
}


std::vector<std::pair<double, double> > car_obs_t::get_control_bounds() const {
    return {
            {0, 2},
            {-.5,.5},
    };
}

std::vector<bool> car_obs_t::is_circular_topology() const {
	return {
			false,
			false,
			true
	};
}


void car_obs_t::normalize(const double* state, double* normalized){
    normalized[STATE_X] = state[STATE_X] / MAX_X;
    normalized[STATE_Y] = state[STATE_Y] /MAX_Y;
    normalized[STATE_THETA] = state[STATE_THETA] / M_PI;
}

void car_obs_t::denormalize(double* normalized, double* state){
    state[STATE_X] = normalized[STATE_X] * MAX_X;
    state[STATE_Y] = normalized[STATE_Y] * MAX_Y;
    state[STATE_THETA] = normalized[STATE_THETA] * M_PI;
}
double car_obs_t::distance(const double* point1, const double* point2, unsigned int state_dimensions){
    double result = 0;
    for (unsigned int i=0; i<state_dimensions; ++i) {
        if (i == 2) {
            double val = fabs(point1[i]-point2[i]);
            if(val > M_PI)
                val = 2*M_PI-val;
            result += val*val;
        } else {
            result += (point1[i]-point2[i]) * (point1[i]-point2[i]);
        }
    }
    return std::sqrt(result);
};

double car_obs_t::get_loss(double* state, const double* goal, double* weight){
    // return angular_error(state[2], goal[2]) * weight[2] + 
    //     abs(state[0] - goal[0]) * weight[0] + abs(state[1] - goal[1]) * weight[1] + abs(state[3] - goal[3]) * weight[3];
    double val = fabs(state[STATE_THETA]-goal[STATE_THETA]);
    if(val > M_PI)
            val = 2*M_PI-val;
    return std::sqrt(val * val * weight[STATE_THETA] + pow(state[STATE_X]-goal[STATE_X], 2.0) * weight[STATE_X]+ pow(state[STATE_Y]-goal[STATE_Y], 2.0)* weight[STATE_Y]);
}