#include <iostream> 
#include <fstream> 
#include <string>
#include <vector>
#include <stdlib.h> 
#include <pybind11/pybind11.h> 
#include "gyems_can_functions.h" 
#include "renishaw_can_functions.hpp" 

#include "model526.h" 
#include "robot_cal.hpp"  

using namespace std; 

#include <unistd.h> 
#include <signal.h>  
#include <cmath> 
#include <stdio.h>

#include <Eigen/Dense>

using namespace Eigen;

#include <pybind11/numpy.h>
namespace py = pybind11; 

// #define PI 3.1415926 
// const double ctl_ratio_1 = -2000.0/32;     
// const double ctl_ratio_2 = 2000.0/32;  
// const double d_t = 0.001; 
// const double L_1 = 0.30, L_2 = 0.25;   

#define STRINGIFY(x) #x 
#define MACRO_STRINGIFY(x) STRINGIFY(x)


void 
sigint_1_step(int dummy) { 
    if (run_on == 1) 
		run_on = 0; 
}


int phri_get_demonstration(
double theta_1_initial, double theta_2_initial,  
double q_1_initial, double q_2_initial,  
double stiffness_1, double stiffness_2,  
double damping_1, double damping_2,  
string angle_path_name, string torque_path_name
{
    ////////////////////////////////////////////
    //// Initial Encoder and Motor CAN 
    //////////////////////////////////////////// 
    CANDevice can0((char *) "can0");   
    can0.begin();   

    CANDevice can1((char *) "can1");   
    can1.begin();   

    Gcan motor_1(can1);   
    Gcan motor_2(can0);   
    motor_1.begin();   
    motor_2.begin();   

    ////////////////////////////////////////////
    // Read original encoder
    ////////////////////////////////////////////
    controller_renishaw encoder("can2");  
    float encoder_arr[2];  

    encoder.read_ang_encoder(encoder_arr);  
  	q_1_t = (double) encoder_arr[1] * PI/180.0 - q_1_initial;   
  	q_2_t = (double) encoder_arr[0] * PI/180.0 - q_2_initial;   
    
    ////////////////////////////////////////////
    // Read original encoder
    ////////////////////////////////////////////
    double theta_1_t = 0.0;   
    double theta_2_t = 0.0;   

    double q_1_t = 0.0;   
    double q_2_t = 0.0;   

    double d_theta_1_t = 0.0;    
    double d_theta_2_t = 0.0;   

    double torque_1 = 0.0;  
    double torque_2 = 0.0;  

    double torque_1_o = 0.0;  
    double torque_2_o = 0.0;  

    double torque_1_e = 0.0;   
    double torque_2_e = 0.0;   

    double torque_1_t = 0.0;   
    double torque_2_t = 0.0;   

    double pos_1 = 0.0;   
    double pos_2 = 0.0;   

    double torque_lower_bound = -0.4;      
    double torque_upper_bound = 0.4;    

    // //////////////////////////////////////////
    // One loop control demonstration
    // //////////////////////////////////////////
    printf("Get demonstration start !!!!\n");  

    string output_torque = torque_path_name;    
    ofstream OutFileTorque(output_torque);    
    OutFileTorque << "torque_1" << "," << "torque_1_t" << "," << "torque_2" << "," << "torque_2_t" << "\n";   

    string output_angle = angle_path_name;    
    ofstream OutFileAngle(output_angle);    
    OutFileAngle << "theta_1_t" << "," << "d_theta_1_t" << "," << "q_1_t" << "," << "theta_2_t" << "," << "d_theta_2_t" << "," << "q_2_t" << "\n";       

    // double stiffness_1 = stiffness_1;    
    // double stiffness_2 = stiffness_2;     

    // double damping_1 = damping_1;     
    // double damping_2 = damping_2;     

    run_on = 1;   

    // Catch a Ctrl-C event:  
    // pointer to signal handler
	void (*sig_h)(int) = sigint_1_step;   

    // py::buffer_info buff_size_list = buff_size.request();   

    // // allocate the output buffer
	// py::array_t<double> result = py::array_t<double>(buff_size_list.size);  

    // result.resize({buff_size_list.shape[0], buff_size_list.shape[1]}); 

    int index = 0;  
    int index_buff = 0;  
    int buff_length = 10000;   

    printf("Print enter for starting record !!!!\n");  
    getchar();  

    ///////////////////////////////////////////////////////
    // avoid large motion at starting points 
    ///////////////////////////////////////////////////////
    for(int index=0; index<10; index=index+1)   
    {
        pos_1 = motor_1.set_torque(motor_id_1, 0.0, &d_theta_1_t, &torque_1_t);   
        pos_2 = motor_2.set_torque(motor_id_2, 0.0, &d_theta_2_t, &torque_2_t);   
    }

    while(run_on)   
    {
        // Catch a Ctrl-C event:   
        signal(SIGINT, sig_h);   
        
        // get sensing data
        theta_1_t = motor_1.read_sensor(motor_id_1) - theta_1_initial;   
        theta_2_t = motor_2.read_sensor(motor_id_2) - theta_2_initial;  

        encoder.read_ang_encoder(encoder_arr);  
    
        q_1_t = (double) encoder_arr[1] * PI/180.0 - q_1_initial;   
        q_2_t = (double) encoder_arr[0] * PI/180.0 - q_2_initial;   

        // printf("torque_1_t: %f\n", torque_1_t);   
        // printf("torque_2_t: %f\n", torque_2_t);   

        // calculate torque data   

        /// zero impedance control   
        torque_1 = clip(stiffness_1 * (torque_1_e - torque_1_t), torque_lower_bound, torque_upper_bound) * ctl_ratio_1;   
        torque_2 = clip(stiffness_2 * (torque_2_e - torque_2_t), torque_lower_bound, torque_upper_bound) * ctl_ratio_2;   

        pos_1 = motor_1.set_torque(motor_id_1, torque_1, &d_theta_1_t, &torque_1_t);    
        pos_2 = motor_2.set_torque(motor_id_2, torque_2, &d_theta_2_t, &torque_2_t);   

        torque_1_o = clip(stiffness_1 * (torque_1_e - torque_1_t), torque_lower_bound, torque_upper_bound);    
        torque_2_o = clip(stiffness_2 * (torque_2_e - torque_2_t), torque_lower_bound, torque_upper_bound);   

        // theta_list[index_buff*result_buf.shape[1] + 0] = theta_1_t;    
        // theta_list[index_buff*result_buf.shape[1] + 1] = theta_2_t;    

        OutFileAngle << theta_1_t << "," << d_theta_1_t << "," << q_1_t << ","<< theta_2_t << "," << d_theta_2_t << ","  << q_2_t <<"\n";  

        OutFileTorque << torque_1_o << "," << torque_1_t << "," << torque_2_o << "," << torque_2_t << "\n";   

        // OutFileVel << d_theta_1_t << " " << d_theta_2_t << "\n";    

        index = index + 1;    
    } 

    OutFileAngle.close();    
    OutFileTorque.close();       

    motor_1.pack_stop_cmd(motor_id_1);   
    motor_2.pack_stop_cmd(motor_id_2);   

    return 1;  
}  



int move_to_target_point(
double stiffness_1, double stiffness_2,  
double damping_1, double damping_2,  
py::array_t<double> q_1_target, py::array_t<double> q_2_target, int N,  
double q_1_initial, double q_2_initial,  
double theta_1_initial, double theta_2_initial,  
double dist_threshold,  
string angle_path_name, string torque_path_name   
)
{
    ////////////////////////////////////////////////////////
    //// Initial Encoder and Motor CAN
    //////////////////////////////////////////////////////// 
    CANDevice can0((char *) "can0");   
    can0.begin();   
    CANDevice can1((char *) "can1");   
    can1.begin();   

    Gcan motor_1(can1);   
    Gcan motor_2(can0);   
    motor_1.begin();   
    motor_2.begin();   

    printf("Move to target point start !!!\n");   

    ////////////////////////////////////////////////////////
    // Initialization of One Loop Control Demonstration ////
    ////////////////////////////////////////////////////////
    string output_angle = torque_path_name;    
    ofstream OutFileAngle(output_angle);    
    OutFileAngle << "angle_1" << "," << "angle_1_t" << "," << "angle_2" << "," << "angle_2_t" << "\n";    

    string output_torque = angle_path_name;   
    ofstream OutFileTorque(output_torque);   
    OutFileTorque << "torque_1" << "," << "torque_1_t" << "," << "torque_2" << "," << "torque_2_t" << "\n";    

    double torque_lower_bound = -1.5;    
    double torque_upper_bound = 1.5;    

    double theta_t_list[2] = {0.0, 0.0};  
    double theta_1_t = 0.0;   
    double theta_2_t = 0.0;   

    double d_theta_t_list[2] = {0.0, 0.0};
    double d_theta_1_t = 0.0;    
    double d_theta_2_t = 0.0;    

    double theta_e_list[2] = {0.0, 0.0}; 
    double theta_1_e = 0.0;   
    double theta_2_e = 0.0;   

    double d_theta_e_list[2] = {0.0, 0.0}; 
    double d_theta_1_e = 0.0;   
    double d_theta_2_e = 0.0;  

    double torque_1 = 0.0;   
    double torque_2 = 0.0;   

    double torque_1_t = 0.0;   
    double torque_2_t = 0.0;   

    double pos_1 = 0.0;      
    double pos_2 = 0.0;      
 
    double dist = 1.0;    

    py::buffer_info q_1_list_buf = q_1_target.request();  
    py::buffer_info q_2_list_buf = q_2_target.request();  
    double *q_1_list = (double *)q_1_list_buf.ptr; 
    double *q_2_list = (double *)q_2_list_buf.ptr;  

    double params[4] = {stiffness_1, stiffness_2, damping_1, damping_2};  
    // params[0] = stiffness_1;  
    // params[1] = stiffness_2;   
    // params[2] = damping_1;    
    // params[3] = damping_2;    

    /////////////////////////////////////////////////////
    /////  avoid large motion at starting points  ///////
    ///////////////////////////////////////////////////// 
    // for(int index=0; index<5; index=index+1)  
    // {
    //     pos_1 = motor_1.set_torque(2, 0.0, &d_theta_1_t, &torque_1_t); 
    //     pos_2 = motor_2.set_torque(1, 0.0, &d_theta_2_t, &torque_2_t); 
    // } 

    run_on = 1;  

    // Catch a Ctrl-C event:
	void  (*sig_h)(int) = sigint_1_step;   // pointer to signal handler
    
    int index = 0;  

    // dist > dist_threshold && initial_index < max_index  
    while(run_on && index<N)   
    {
        // Catch a Ctrl-C event:    
        signal(SIGINT, sig_h);    

        // theta_1_t = motor_1.read_sensor(2) - theta_1_initial;  
        // theta_2_t = -1 * (motor_2.read_sensor(1) + theta_1_t - theta_2_initial);   
        theta_1_t = motor_1.read_sensor(motor_id_1) - theta_1_initial;   
        theta_2_t = motor_2.read_sensor(motor_id_2) - theta_2_initial;   

        dist = sqrt(pow((theta_1_t - q_1_list[index]), 2) + pow((theta_2_t - q_2_list[index]), 2));    
        // printf("theta_1_t: %f\n", theta_1_t);     
        // printf("theta_2_t: %f\n", theta_2_t);    

        theta_t_list[0] = theta_1_t;
        theta_t_list[1] = theta_2_t;   

        theta_e_list[0] = q_1_list[index]; 
        theta_e_list[1] = q_2_list[index]; 

        d_theta_t_list[0] = d_theta_1_t;   
        d_theta_t_list[1] = d_theta_2_t;   

        /////////////////////////////////////////////////////
        // calculate torque control command 
        ///////////////////////////////////////////////////// 
        torque_calculation(
        params, 
        theta_e_list, d_theta_e_list, theta_t_list, d_theta_t_list,  
        torque_lower_bound, torque_upper_bound,  
        torque_1, torque_2   
        );  

        // torque_1 = clip(-1 * stiffness_1 * (q_1_list[index] - theta_1_t) - damping_1 * (d_theta_1_e - d_theta_1_t), torque_lower_bound, torque_upper_bound) * ctl_ratio_1; 
        // torque_2 = clip(-1 * stiffness_2 * (q_2_list[index] - theta_2_t) - damping_2 * (d_theta_2_e - d_theta_2_t), torque_lower_bound, torque_upper_bound) * ctl_ratio_2; 

        double torque_1_o = -1 * stiffness_1 * (q_1_list[index] - theta_1_t) - damping_1 * (d_theta_1_e - d_theta_1_t);  
        double torque_2_o = -1 * stiffness_2 * (q_2_list[index] - theta_2_t) - damping_2 * (d_theta_2_e - d_theta_2_t);  

        // OutFileAngle << theta_1_t << "," << theta_2_t << "\n";  
        OutFileAngle << q_1_list[index] << "," << theta_1_t << "," << q_2_list[index] << "," << theta_2_t << "\n";  

        pos_1 = motor_1.set_torque(motor_id_1, torque_1, &d_theta_1_t, &torque_1_t);    
        pos_2 = motor_2.set_torque(motor_id_2, torque_2, &d_theta_2_t, &torque_2_t);    

        OutFileTorque << torque_1_o << "," << torque_1 << "," << torque_2_o << "," << torque_2 << "\n";   

        // OutFileVel << d_theta_1_t << " " << d_theta_2_t << "\n";   

        // printf("d_theta_1_t: %f\n", d_theta_1_t);   
        // printf("d_theta_2_t: %f\n", d_theta_2_t);   
        index = index + 1;  
    } 

    OutFileAngle.close();   
    OutFileTorque.close();       

    motor_1.pack_stop_cmd(2);   
    motor_2.pack_stop_cmd(1);   

    // printf("Move to target point done !!!! \n");   

    return 1;  
}