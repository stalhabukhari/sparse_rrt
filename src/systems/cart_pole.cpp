/**
 * @file cart_pole.cpp
 * 
 * @copyright Software License Agreement (BSD License)
 * Copyright (c) 2014, Rutgers the State University of New Jersey, New Brunswick  
 * All Rights Reserved.
 * For a full description see the file named LICENSE.
 * 
 * Authors: Zakary Littlefield, Kostas Bekris 
 * 
 */


#include "systems/cart_pole.hpp"
#include "utilities/random.hpp"


#define _USE_MATH_DEFINES


#include <cmath>


#define I 10
#define L 2.5
#define M 10
#define m 5
#define g 9.8

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


bool cart_pole_t::propagate( double* start_state, double* control, int num_steps, double* result_state, double integration_step)
{
        temp_state[0] = start_state[0]; 
        temp_state[1] = start_state[1];
        temp_state[2] = start_state[2];
        temp_state[3] = start_state[3];
        bool validity = true;
        for(int i=0;i<num_steps;i++)
        {
                update_derivative(control);
                temp_state[0] += integration_step*deriv[0];
                temp_state[1] += integration_step*deriv[1];
                temp_state[2] += integration_step*deriv[2];
                temp_state[3] += integration_step*deriv[3];
                enforce_bounds();
                validity = validity && valid_state();
        }
        result_state[0] = temp_state[0];
        result_state[1] = temp_state[1];
        result_state[2] = temp_state[2];
        result_state[3] = temp_state[3];
        return validity;
}

void cart_pole_t::enforce_bounds()
{
        if(temp_state[0]<MIN_X)
                temp_state[0]=MIN_X;
        else if(temp_state[0]>MAX_X)
                temp_state[0]=MAX_X;

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


bool cart_pole_t::valid_state()
{
    return true;
}

svg::Point cart_pole_t::visualize_point(const double* state, svg::Dimensions dims)
{
        double x = state[STATE_X] + (L / 2.0) * sin(state[STATE_THETA]);
        double y = -(L / 2.0) * cos(state[STATE_THETA]);

        x = (x-MIN_X)/(MAX_X-MIN_X) * dims.width; 
        // y = (y+L)/(2*L) * dims.height; 
        y = (y-MIN_X)/(MAX_X-MIN_X) * dims.height; 
        return svg::Point(x,y);
}

void cart_pole_t::update_derivative(double* control)
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


std::vector<std::pair<double, double> > cart_pole_t::get_state_bounds() {
    return {
            {MIN_X,MAX_X},
            {MIN_V,MAX_V},
            {-M_PI,M_PI},
            {MIN_W,MAX_W},
    };
}


std::vector<std::pair<double, double> > cart_pole_t::get_control_bounds() {
    return {
            {-300,300},
    };
}


std::vector<bool> cart_pole_t::is_circular_topology() {
    return {
            false,
            false,
            true,
            false
    };
}