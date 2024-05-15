/**
 * @file cart_pole_obs.cpp
 *
 * @copyright Software License Agreement (BSD License)
 * Original work Copyright (c) 2014, Rutgers the State University of New Jersey, New Brunswick
 * Modified work Copyright 2017 Oleg Y. Sinyavskiy
 * All Rights Reserved.
 * For a full description see the file named LICENSE.
 *
 * Original authors: Zakary Littlefield, Kostas Bekris
 * Modifications by: Yinglong Miao
 *
 */


#include "systems/cart_pole_obs.hpp"
#include "utilities/random.hpp"
#include <iostream>

#define _USE_MATH_DEFINES


#include <cmath>


#define I 10
#define L 2.5
#define M 10
#define m 5
#define g 9.8
// height of the cart
#define H 0.5

#define STATE_X 0
#define STATE_V 1
#define STATE_THETA 2
#define STATE_W 3
#define CONTROL_A 0

#define MIN_X -30
#define MAX_X 30
#define MIN_V -40
#define MAX_V 40
#define MIN_W -2
#define MAX_W 2


bool cart_pole_obs_t::propagate(
    const double* start_state, unsigned int state_dimension,
    const double* control, unsigned int control_dimension,
    int num_steps, double* result_state, double integration_step)
{
        // std::cout<< start_state[0] <<","<<  start_state[1] <<","<<  start_state[2] <<","<<  start_state[3]<<std::endl;
        temp_state[0] = start_state[0];
        temp_state[1] = start_state[1];
        temp_state[2] = start_state[2];
        temp_state[3] = start_state[3];
        bool validity = false;
        double enforced_control;
        if(*control > 300){
            enforced_control = 300;
        } else if(*control < -300){
            enforced_control = -300;
        } else {
            enforced_control = *control;
        }
        for(int i=0;i<num_steps;i++)
        {
                update_derivative(&enforced_control);
                temp_state[0] += integration_step*deriv[0];
                temp_state[1] += integration_step*deriv[1];
                temp_state[2] += integration_step*deriv[2];
                temp_state[3] += integration_step*deriv[3];
                enforce_bounds();
                //validity = validity && valid_state();
                if (valid_state() == true)
                {
                    result_state[0] = temp_state[0];
                    result_state[1] = temp_state[1];
                    result_state[2] = temp_state[2];
                    result_state[3] = temp_state[3];
                    validity = true;
                }
                else
                {
                    // Found the earliest invalid position. break the loop and return
                    validity = false; // need to update validity because one node is invalid, the propagation fails
                    break;
                }
        }
        // std::cout<< result_state[0] <<","<<  result_state[1] <<","<<  result_state[2] <<","<<  result_state[3]<<std::endl;
        // std::cout<<validity<<std::endl;
        // std::cout<<validity<<std::endl;
        //result_state[0] = temp_state[0];
        //result_state[1] = temp_state[1];
        //result_state[2] = temp_state[2];
        //result_state[3] = temp_state[3];
        return validity;
}

void cart_pole_obs_t::enforce_bounds()
{
        // fpr the position, if it is outside of bound, we don't enforce it back
        //if(temp_state[0]<MIN_X)
        //        temp_state[0]=MIN_X;
        //else if(temp_state[0]>MAX_X)
        //        temp_state[0]=MAX_X;

        if(temp_state[1]<MIN_V)
                temp_state[1]=MIN_V;
        else if(temp_state[1]>MAX_V)
                temp_state[1]=MAX_V;

        if(temp_state[2]<-M_PI)
                temp_state[2]+=2*M_PI;
        else if(temp_state[2]>M_PI)
                temp_state[2]-=2*M_PI;

        if(temp_state[3]<MIN_W)
                temp_state[3]=MIN_W;
        else if(temp_state[3]>MAX_W)
                temp_state[3]=MAX_W;
}


bool cart_pole_obs_t::valid_state()
{
    // check the pole with the rectangle to see if in collision
    // calculate the pole state
    // check if the position is within bound
    if (temp_state[0] < MIN_X or temp_state[0] > MAX_X)
    {
        return false;
    }
    double pole_x1 = temp_state[0];
    double pole_y1 = H;
    double pole_x2 = temp_state[0] + L * sin(temp_state[2]);
    double pole_y2 = H + L * cos(temp_state[2]);
    //std::cout << "state:" << temp_state[0] << "\n";
    //std::cout << "pole point 1: " << "(" << pole_x1 << ", " << pole_y1 << ")\n";
    //std::cout << "pole point 2: " << "(" << pole_x2 << ", " << pole_y2 << ")\n";
    for(unsigned int i = 0; i < obs_list.size(); i++)
    {
        // check if any obstacle has intersection with pole
        //std::cout << "obstacle " << i << "\n";
        //std::cout << "points: \n";
        for (unsigned int j = 0; j < 8; j += 2)
        {
            // check each line of the obstacle
            double x1 = obs_list[i][j];
            double y1 = obs_list[i][j+1];
            double x2 = obs_list[i][(j+2) % 8];
            double y2 = obs_list[i][(j+3) % 8];
            if (lineLine(pole_x1, pole_y1, pole_x2, pole_y2, x1, y1, x2, y2))
            {
                // intersect
                return false;
            }
        }
    }
    return true;
}

std::tuple<double, double> cart_pole_obs_t::visualize_point(const double* state, unsigned int state_dimension) const
{
    double x = state[STATE_X] + (L / 2.0) * sin(state[STATE_THETA]);
    double y = -(L / 2.0) * cos(state[STATE_THETA]);

    x = (x-MIN_X)/(MAX_X-MIN_X);
    y = (y-MIN_X)/(MAX_X-MIN_X);
    return std::make_tuple(x, y);
}

void cart_pole_obs_t::update_derivative(const double* control)
{
    double _v = temp_state[STATE_V];
    double _w = temp_state[STATE_W];
    double _theta = temp_state[STATE_THETA];
    double _a = control[CONTROL_A];
    double mass_term = (M + m)*(I + m * L * L) - m * m * L * L * cos(_theta) * cos(_theta);

    deriv[STATE_X] = _v;
    deriv[STATE_THETA] = _w;
    mass_term = (1.0 / mass_term);
    deriv[STATE_V] = ((I + m * L * L)*(_a + m * L * _w * _w * sin(_theta)) + m * m * L * L * cos(_theta) * sin(_theta) * g) * mass_term;
    deriv[STATE_W] = ((-m * L * cos(_theta))*(_a + m * L * _w * _w * sin(_theta))+(M + m)*(-m * g * L * sin(_theta))) * mass_term;
}


bool cart_pole_obs_t::lineLine(double x1, double y1, double x2, double y2, double x3, double y3, double x4, double y4)
// compute whether two lines intersect with each other
{
    // ref: http://www.jeffreythompson.org/collision-detection/line-rect.php
    // calculate the direction of the lines
    double uA = ((x4-x3)*(y1-y3) - (y4-y3)*(x1-x3)) / ((y4-y3)*(x2-x1) - (x4-x3)*(y2-y1));
    double uB = ((x2-x1)*(y1-y3) - (y2-y1)*(x1-x3)) / ((y4-y3)*(x2-x1) - (x4-x3)*(y2-y1));

    // if uA and uB are between 0-1, lines are colliding
    if (uA >= 0 && uA <= 1 && uB >= 0 && uB <= 1)
    {
        // intersect
        return true;
    }
    // not intersect
    return false;
}



std::vector<std::pair<double, double> > cart_pole_obs_t::get_state_bounds() const {
    return {
            {MIN_X,MAX_X},
            {MIN_V,MAX_V},
            {-M_PI,M_PI},
            {MIN_W,MAX_W},
    };
}


std::vector<std::pair<double, double> > cart_pole_obs_t::get_control_bounds() const {
    return {
            {-300,300},
    };
}


std::vector<bool> cart_pole_obs_t::is_circular_topology() const {
    return {
            false,
            false,
            true,
            false
    };
}


double cart_pole_obs_t::get_loss(double* state, const double* goal, double* weight){
    // return angular_error(state[2], goal[2]) * weight[2] + 
    //     abs(state[0] - goal[0]) * weight[0] + abs(state[1] - goal[1]) * weight[1] + abs(state[3] - goal[3]) * weight[3];
    double val = fabs(state[STATE_THETA]-goal[STATE_THETA]);
    if(val > M_PI)
            val = 2*M_PI-val;
    return std::sqrt(val * val * weight[STATE_THETA] + pow(state[STATE_X]-goal[STATE_X], 2.0) * weight[STATE_X]+ pow(state[STATE_V]-goal[STATE_V], 2.0)* weight[STATE_V] 
        + pow(state[STATE_W]-goal[STATE_W], 2.0)* weight[STATE_W]);
}

double cart_pole_obs_t::angular_error(double angle, double goal){
    double error = angle - goal;
    if(error < 0){
        error = - error;
    }
    else if (error > M_PI){
        error = 2*M_PI - error;
    }
    return error;
}


void cart_pole_obs_t::normalize(const double* state, double* normalized){
    normalized[STATE_X] = state[STATE_X] / MAX_X;
    normalized[STATE_V] = state[STATE_V] /MAX_V;
    normalized[STATE_THETA] = state[STATE_THETA] / M_PI;
    normalized[STATE_W] = state[STATE_W] / MAX_W;
}

void cart_pole_obs_t::denormalize(double* normalized, double* state){
    state[STATE_X] = normalized[STATE_X] * MAX_X;
    state[STATE_V] = normalized[STATE_V] * MAX_V;
    state[STATE_THETA] = normalized[STATE_THETA] * M_PI;
    state[STATE_W] = normalized[STATE_W]  * MAX_W;
}

double cart_pole_obs_t::distance(const double* point1, const double* point2, unsigned int)
{
    double val = fabs(point1[STATE_THETA]-point2[STATE_THETA]);
    if(val > M_PI)
            val = 2*M_PI-val;
    return std::sqrt(val * val + pow(point1[0]-point2[0], 2.0) + pow(point1[1]-point2[1], 2.0)+ pow(point1[3]-point2[3], 2.0));
    // return std::sqrt( val * val + pow(point1[0]-point2[0], 2.0)  );

}

// double cart_pole_obs_t::state_distance(const double* point1, const double* point2, unsigned int)
// {
//     double val = fabs(point1[STATE_THETA]-point2[STATE_THETA]);
//     if(val > M_PI)
//             val = 2*M_PI-val;
//     return std::sqrt(val * val + pow(point1[0]-point2[0], 2.0) + pow(point1[1]-point2[1], 2.0)+ pow(point1[3]-point2[3], 2.0));
//     // return std::sqrt( val * val + pow(point1[0]-point2[0], 2.0)  );

// }