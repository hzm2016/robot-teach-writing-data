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
#include "SEA_model.h"

using namespace std; 

#include <unistd.h> 
#include <signal.h>  
#include <cmath> 
#include <stdio.h>

#include <Eigen/Dense>

using namespace Eigen; 

#include <pybind11/numpy.h>
namespace py = pybind11; 


#define STRINGIFY(x) #x 
#define MACRO_STRINGIFY(x) STRINGIFY(x)


int run_on; 

void 
sigint_1_step(int dummy) { 
    if (run_on == 1) 
		run_on = 0; 
}


double set_two_link_position(
double theta_1_initial, double theta_2_initial, 
int32_t angle_1, int32_t angle_2
)   
{
    ////////////////////////////////////////////
    // Read motor angle 3  
    //////////////////////////////////////////// 
    CANDevice can1((char *) "can1");  
    can1.begin();  

    Gcan motor_1(can1);   
    motor_1.begin();  
    
    double theta_1 = motor_1.read_sensor(2) - theta_1_initial;  

    CANDevice can0((char *) "can0");   
    can0.begin();   

    Gcan motor_2(can0);   
    motor_2.begin();   

    double theta_2 = -1 * (motor_2.read_sensor(1) + theta_1 - theta_2_initial);   
    printf("Motor 1 position: %f%f\n", theta_1, theta_1/PI*180);   
    printf("Motor 2 position: %f%f\n", theta_2, theta_2/PI*180);    
    // printf("Motor 3 position: %f\n", theta_3_t/3.14*180);   

    // int32_t angle = theta_3_t + 10; 
    uint16_t max_speed = 20; 
    int32_t angle_fixed = 200;  

    motor_1.pack_position_2_cmd(2, angle_1, max_speed);  
    motor_2.pack_position_2_cmd(1, angle_2, max_speed);  

    // motor_3.read_encoder(1); 
    // motor_3.readcan(); 

    run_on = 1;  
    // Catch a Ctrl-C event:
	void  (*sig_h)(int) = sigint_1_step;   // pointer to signal handler

    // while(run_on) 
    // {
    //     // Catch a Ctrl-C event: 
    //     signal(SIGINT, sig_h);  

    //     motor_3.set_torque(1, 50, &d_theta_1_t, &torque_1_t); 
    // }
    
    // motor_3.pack_stop_cmd(1); 

    return 1; 
}


int set_position(double theta_3_initial, int32_t angle)   
{
    ////////////////////////////////////////////
    // Read motor angle 3 
    ////////////////////////////////////////////
    CANDevice can0((char *) "can2");   
    can0.begin();   

    Gcan motor_3(can0);   
    motor_3.begin();   
 
    double theta_3_t=0.0;   

    // theta_3_t = motor_3.read_single_turn(1);  
    // printf("Motor 3 position: %f\n", theta_3_t/3.14*180);   

    // int32_t angle = theta_3_t + 10; 
    uint16_t max_speed = 20;   
    int32_t angle_fixed = 200;   

    motor_3.pack_position_2_cmd(1, angle, max_speed);   
    // motor_3.pack_position_1_cmd(1, angle_fixed); 

    // motor_3.read_encoder(1);   
    // motor_3.readcan();  

    run_on = 1;   
    // Catch a Ctrl-C event:  
	void  (*sig_h)(int) = sigint_1_step;   // pointer to signal handler

    // while(run_on) 
    // {
    //     // Catch a Ctrl-C event: 
    //     signal(SIGINT, sig_h);  

    //     motor_3.set_torque(1, 50, &d_theta_1_t, &torque_1_t); 
    // } 
    
    // motor_3.pack_stop_cmd(1);  

    return 1;   
}  


void Cal_torque(double theta_1_t, double theta_2_t, double F_1_t, double F_2_t) 
{
    // const <MatrixXd> J 
    MatrixXd m(2,2);  
    Vector2d F_t(F_1_t, F_2_t);  
    Vector2d tau_t(0.0, 0.0);   
    Vector2d pos_t(0.0, 0.0);   

    // m(0, 0) = - L_1 * sin(theta_1_t) - L_2 * sin(theta_1_t + theta_2_t);  
    // m(0, 1) = - L_2 * sin(theta_1_t + theta_2_t);  
    // m(1, 0) = L_1 * cos(theta_1_t) + L_2 * cos(theta_1_t + theta_2_t);   
    // m(1, 1) = L_2 * cos(theta_1_t + theta_2_t);  
    m = Jacobian(theta_1_t, theta_2_t);   
    pos_t = Forward_ik(theta_1_t, theta_2_t);   

    tau_t = m.transpose() * F_t;   

    printf("matrix :%f%f%f%f\n", m(0,0), m(0,1), m(1, 0), m(1, 1));   
    printf("vector :%f%f\n", tau_t(0), tau_t(1));   
    printf("vector :%f%f\n", pos_t(0), pos_t(1));   

    // return m; 
}


void Convert_stiffness(
double stiffness_1_t, double stiffness_2_t, 
double theta_1_t, double theta_2_t
)
{
    MatrixXd J_t(2,2);    
    Vector2d stiff_joint_t(0.0, 0.0);     
    MatrixXd stiff_task_t(2, 2);     
    stiff_joint_t(0) = stiffness_1_t;     
    stiff_joint_t(1) = stiffness_2_t;     

    MatrixXd Stiff_Joint_Diag(stiff_joint_t.asDiagonal());    
    J_t = Jacobian(theta_1_t, theta_2_t);     

    stiff_task_t = J_t.transpose() * Stiff_Joint_Diag * J_t;    
    printf("stiffness %f %f\n:", stiff_task_t(0, 0), stiff_task_t(1, 1));    
}


double calculate_joint_torque(
double params[4],  
double theta_e_list[2], double d_theta_e_list[2],   
double theta_t_list[2], double d_theta_t_list[2],   
double &torque_1, double &torque_2   
) 
{
    /////////////////////////////////////////////////////
    // calculate torque control command 
    ///////////////////////////////////////////////////// 
    MatrixXd J_t(2,2);    
    MatrixXd stiff_joint_t(2, 2);    
    MatrixXd damping_joint_t(2, 2);    

    Vector2d stiff_task_t(0.0, 0.0);      
    stiff_task_t(0) = params[0];     
    stiff_task_t(1) = params[1];    

    Vector2d damping_task_t(0.0, 0.0);   
    damping_task_t(0) = params[2];     
    damping_task_t(1) = params[3];  

    Vector2d torque_t(0.0, 0.0);  
    Vector2d delta_angle_t(0.0, 0.0); 
    Vector2d d_delta_angle_t(0.0, 0.0); 
    delta_angle_t(0) =  theta_t_list[0] - theta_e_list[0]; 
    delta_angle_t(1) =  theta_t_list[1] - theta_e_list[1];  
    d_delta_angle_t(0) =  d_theta_t_list[0] - d_theta_e_list[0]; 
    delta_angle_t(1) =  d_theta_t_list[1] - d_theta_e_list[1];  

    MatrixXd Stiff_Task_Diag(stiff_task_t.asDiagonal());    
    MatrixXd Damping_Task_Diag(damping_task_t.asDiagonal());    
    J_t = Jacobian(theta_t_list[0], theta_t_list[1]);     

    stiff_joint_t = J_t.transpose() * Stiff_Task_Diag * J_t;    
    damping_joint_t = J_t.transpose() * Damping_Task_Diag * J_t;    

    printf("stiffness %f - %f\n:", stiff_joint_t(0, 0), stiff_joint_t(1, 1));    

    torque_t = - stiff_joint_t * delta_angle_t - damping_joint_t * d_delta_angle_t;  
    printf("torque_t %f - %f\n:", torque_t(0), torque_t(1)); 

    torque_1 = clip(-1 * params[0] * (theta_t_list[0] - theta_e_list[0]) - params[2] * (d_theta_t_list[0] - d_theta_e_list[0]), torque_lower_bound, torque_upper_bound) * ctl_ratio_1; 
    torque_2 = clip(-1 * params[1] * (theta_t_list[1] - theta_e_list[1]) - params[3] * (d_theta_t_list[1] - d_theta_e_list[1]), torque_lower_bound, torque_upper_bound) * ctl_ratio_2;     

    // torque_1 = clip(torque_t(0), torque_lower_bound, torque_upper_bound) * ctl_ratio_1;   
    // torque_2 = clip(torque_t(1), torque_lower_bound, torque_upper_bound) * ctl_ratio_2;   
}   


double calculate_task_torque( 
double params[4],  
double theta_e_list[2], double d_theta_e_list[2],  
double theta_t_list[2], double d_theta_t_list[2],  
double &torque_1, double &torque_2  
)
{    
    /////////////////////////////////////////////////////
    // calculate torque control command 
    ///////////////////////////////////////////////////// 
    Vector2d pos_t(0.0, 0.0);   
    Vector2d pos_e(0.0, 0.0);   

    Vector2d d_pos_t(0.0, 0.0);   
    Vector2d d_pos_e(0.0, 0.0);   

    Vector2d d_angle_t(theta_t_list[0], theta_t_list[1]);   
    Vector2d d_angle_e(theta_e_list[0], theta_e_list[1]);   

    MatrixXd J_t(2,2), J_e(2,2); 

    Vector2d F_t(0.0, 0.0); 
    Vector2d torque_t(0.0, 0.0);  

    J_e = Jacobian(theta_e_list[0], theta_e_list[1]);  
    d_pos_e = J_e * d_angle_e;  

    J_t = Jacobian(theta_t_list[0], theta_t_list[1]);  
    d_pos_t = J_t * d_angle_t;  

    pos_e = Forward_ik(theta_e_list[0], theta_e_list[1]);  
    pos_t = Forward_ik(theta_t_list[0], theta_t_list[1]);  

    F_t(0) = - params[0] * (pos_t(0) - pos_e(0)) - params[2] * (d_pos_t(0) - d_pos_e(0));   
    F_t(1) = - params[1] * (pos_t(1) - pos_e(1)) - params[3] * (d_pos_t(1) - d_pos_e(1));   

    torque_t = J_t.transpose() * F_t;   

    torque_1 = clip(torque_t(0), torque_lower_bound, torque_upper_bound) * ctl_ratio_1;   
    torque_2 = clip(torque_t(1), torque_lower_bound, torque_upper_bound) * ctl_ratio_2;   
}


int get_demonstration(
double theta_1_initial, double theta_2_initial,   
double stiffness_1, double stiffness_2,   
double damping_1, double damping_2,   
string angle_name, string torque_name   
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

    printf("Get demonstration start !!!!\n");  

    ////////////////////////////////////////////////////////   
    // One loop control demonstration
    ////////////////////////////////////////////////////////   
    // string output_torque = "demonstrated_torque_list.txt";   
    string output_torque = torque_name;   
    ofstream OutFileTorque(output_torque);    
    OutFileTorque << "torque_1" << "," << "torque_1_t" << "," << "torque_2" << "," << "torque_2_t" << "\n";   

    // string output_angle = "demonstrated_angle_list.txt";   
     string output_angle = angle_name;   
    ofstream OutFileAngle(output_angle);    
    OutFileAngle << "angle_1" << "," << "velocity_1" << " " << "angle_2" << "," << "velocity_2" << "\n";      

    double theta_1_t = 0.0;   
    double theta_2_t = 0.0;   

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

    run_on = 1;   

    // Catch a Ctrl-C event:  
    // pointer to signal handler
	void (*sig_h)(int) = sigint_1_step;   

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
        
        theta_1_t = motor_1.read_sensor(motor_id_1) - theta_1_initial;    
        theta_2_t = motor_2.read_sensor(motor_id_2) - theta_2_initial;    
        // theta_2_t = -1 * (motor_2.read_sensor(motor_id_2) + theta_1_t - theta_2_initial);   

        /// zero impedance control   
        torque_1 = clip(stiffness_1 * (torque_1_e - torque_1_t), torque_lower_bound, torque_upper_bound) * ctl_ratio_1;   
        torque_2 = clip(stiffness_2 * (torque_2_e - torque_2_t), torque_lower_bound, torque_upper_bound) * ctl_ratio_2;   

        // pos_1 = motor_1.set_torque(2, torque_1, &d_theta_1_t, &torque_1_t);    
        // pos_2 = motor_2.set_torque(1, torque_2, &d_theta_2_t, &torque_2_t);    

        pos_1 = motor_1.set_torque(motor_id_1, 0.0, &d_theta_1_t, &torque_1_t);   
        pos_2 = motor_2.set_torque(motor_id_2, 0.0, &d_theta_2_t, &torque_2_t);   

        torque_1_o = clip(stiffness_1 * (torque_1_e - torque_1_t), torque_lower_bound, torque_upper_bound);    
        torque_2_o = clip(stiffness_2 * (torque_2_e - torque_2_t), torque_lower_bound, torque_upper_bound);   

        OutFileAngle << theta_1_t << "," << d_theta_1_t << "," << theta_2_t << "," << d_theta_2_t << "\n";  
        OutFileTorque << torque_1_o << "," << torque_1_t << "," << torque_2_o << "," << torque_2_t << "\n";   

        printf("d_theta_1_t: %f\n", d_theta_1_t);   
        printf("d_theta_2_t: %f\n", d_theta_2_t);   
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
double theta_1_initial, double theta_2_initial,  
double dist_threshold   
)
{
    ////////////////////////////////////////////////////////
    ////////// Initial Encoder and Motor CAN ///////////////
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
    //////////// One loop control demonstration ////////////
    ////////////////////////////////////////////////////////
    string output_angle = "./data/move_target_angle_list.txt";    
    ofstream OutFileAngle(output_angle);    
    OutFileAngle << "angle_1" << "," << "angle_1_t" << "," << "d_angle_1_t" << "," << "angle_2" << "," << "angle_2_t" << "," << "d_angle_2_t"<< "\n";    

    string output_torque = "./data/move_target_torque_list.txt";    
    ofstream OutFileTorque(output_torque);    
    OutFileTorque << "torque_1" << "," << "torque_1_t" << "," << "torque_2" << "," << "torque_2_t" << "\n";    

    ////////////////////////////////////////////////////////
    /////////////////// Hyper-parameters ///////////////////
    ////////////////////////////////////////////////////////
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
    double d_theta_1_old = 0.0;   
    double d_theta_2_old = 0.0;   

    double torque_1 = 0.0;   
    double torque_2 = 0.0;   

    double torque_1_t = 0.0;   
    double torque_2_t = 0.0;   

    double torque_1_o = 0.0;  
    double torque_2_o = 0.0;  

    double pos_1 = 0.0;      
    double pos_2 = 0.0;      
 
    double dist = 1.0;   

    py::buffer_info q_1_list_buf = q_1_target.request();   
    py::buffer_info q_2_list_buf = q_2_target.request();   
    double *q_1_list = (double *)q_1_list_buf.ptr;   
    double *q_2_list = (double *)q_2_list_buf.ptr;   

    double params[4] = {stiffness_1, stiffness_2, damping_1, damping_2};  

    /////////////////////////////////////////////////////
    /////  avoid large motion at starting points  ///////
    ///////////////////////////////////////////////////// 
    // for(int index=0; index<5; index=index+1)  
    // {
    //     pos_1 = motor_1.set_torque(motor_id_1, 0.0, &d_theta_1_t, &torque_1_t);  
    //     pos_2 = motor_2.set_torque(motor_id_2, 0.0, &d_theta_2_t, &torque_2_t);  
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

        theta_1_t = motor_1.read_sensor(motor_id_1) - theta_1_initial;  
        theta_2_t = -1 * (motor_2.read_sensor(motor_id_2) + theta_1_t - theta_2_initial);   

        // dist = sqrt(pow((theta_1_t - q_1_list[index]), 2) + pow((theta_2_t - q_2_list[index]), 2));    

        theta_t_list[0] = theta_1_t;
        theta_t_list[1] = theta_2_t; 

        theta_e_list[0] = q_1_list[index]; 
        theta_e_list[1] = q_2_list[index]; 

        d_theta_t_list[0] = d_theta_1_t; 
        d_theta_t_list[1] = d_theta_2_t; 

        /////////////////////////////////////////////////////
        /////// calculate torque control command //////////// 
        ///////////////////////////////////////////////////// 
        calculate_joint_torque(
        params,  
        theta_e_list, d_theta_e_list,  
        theta_t_list, d_theta_t_list,  
        torque_1, torque_2  
        );  

        /// Task space control
        // calculate_task_torque( 
        // params,  
        // theta_e_list, d_theta_e_list,  
        // theta_t_list, d_theta_t_list,  
        // torque_1, torque_2  
        // );  

        double torque_1_o = torque_1/ctl_ratio_1;   
        double torque_2_o = torque_2/ctl_ratio_2;   

        pos_1 = motor_1.set_torque(motor_id_1, torque_1, &d_theta_1_t, &torque_1_t);    
        pos_2 = motor_2.set_torque(motor_id_2, torque_2, &d_theta_2_t, &torque_2_t);    

        d_theta_1_t = filter(d_theta_1_old, d_theta_1_t);   
        d_theta_2_t = filter(d_theta_2_old, d_theta_1_t);   

        d_theta_1_old = d_theta_1_t;    
        d_theta_2_old = d_theta_2_t;   

        OutFileAngle << q_1_list[index] << "," << theta_1_t << "," << d_theta_1_t << "," << q_2_list[index] << "," << theta_2_t << "," << d_theta_2_t << "\n";  
        OutFileTorque << torque_1_o << "," << torque_1 << "," << torque_2_o << "," << torque_2 << "\n";    
        printf("d_theta_1_t: %f\n", d_theta_1_t);   
        printf("d_theta_2_t: %f\n", d_theta_2_t);   

        index = index + 1;  
    } 

    OutFileAngle.close();   
    OutFileTorque.close();    

    motor_1.pack_stop_cmd(motor_id_1);   
    motor_2.pack_stop_cmd(motor_id_2);   

    printf("Move to target point done !!!! \n");   
    return 1;  
}


int run_one_loop(  
py::array_t<double> theta_1_target, py::array_t<double> theta_2_target,  
py::array_t<double> stiff_1_target, py::array_t<double> stiff_2_target,  
py::array_t<double> damping_1_target, py::array_t<double> damping_2_target,  
int Num_waypoints,  
double theta_1_initial, double theta_2_initial,  
int num_episodes,  
string angle_path_name, string torque_path_name  
)  
{
    ////////////////////////////////////////////////////////
    // Initial hardware ::: can device
    ////////////////////////////////////////////////////////
    CANDevice can0((char *) "can0");
    can0.begin();
    CANDevice can1((char *) "can1");
    can1.begin();

    Gcan motor_1(can1);   
    Gcan motor_2(can0);   
    motor_1.begin();   
    motor_2.begin();   

    std::cout << "Run One Loop !!!" << std::endl;   

    ////////////////////////////////////////////////////////
    // Define file to store data
    ////////////////////////////////////////////////////////  
    ofstream OutFileTorque(torque_path_name);    
    OutFileTorque << "torque_1" << "," << "torque_1_t" << "," << "torque_2" << "," << "torque_2_t" << "\n";  
 
    ofstream OutFileAngle(angle_path_name);    
    OutFileAngle << "angle_1_e" << "," << "angle_1_t" << "," << "d_theta_1_t" << "," << "angle_2_e" << "," << "angle_2_t" << "," << "d_theta_2_t" << "\n";   

    ////////////////////////////////////////////////////////
    // Impedance Parameters 
    ////////////////////////////////////////////////////////
    double params[4] = {30, 20, 0.0, 0.0};   
    
    double theta_t_list[2] = {0.0, 0.0};   
    double theta_1_t = 0.0;   
    double theta_2_t = 0.0;   

    double d_theta_t_list[2] = {0.0, 0.0};   
    double d_theta_1_t = 0.0;   
    double d_theta_2_t = 0.0;   
    double d_theta_1_old = 0.0;
    double d_theta_2_old = 0.0;

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

    double torque_1_o = 0.0; 
    double torque_2_o = 0.0; 

    double pos_1 = 0.0;   
    double pos_2 = 0.0;   

    //////////////////////////////////////////////////////
    // Input data 
    //////////////////////////////////////////////////////
    py::buffer_info theta_1_list_buf = theta_1_target.request();   
    py::buffer_info theta_2_list_buf = theta_2_target.request();   
    double *theta_1_list = (double *)theta_1_list_buf.ptr;   
    double *theta_2_list = (double *)theta_2_list_buf.ptr;    

    py::buffer_info stiff_1_list_buf = stiff_1_target.request();   
    py::buffer_info stiff_2_list_buf = stiff_2_target.request();   
    double *stiff_1_list = (double *)stiff_1_list_buf.ptr;   
    double *stiff_2_list = (double *)stiff_2_list_buf.ptr;    

    py::buffer_info damping_1_list_buf = damping_1_target.request();   
    py::buffer_info damping_2_list_buf = damping_2_target.request();   
    double *damping_1_list = (double *)damping_1_list_buf.ptr;   
    double *damping_2_list = (double *)damping_2_list_buf.ptr;    

	py::array_t<double> result = py::array_t<double>(theta_1_list_buf.size); 

	py::buffer_info result_buf = result.request();  

    double *return_list = (double *)result_buf.ptr;   
    
    run_on = 1; 

    // Catch a Ctrl-C event:
	void  (*sig_h)(int) = sigint_1_step;   // pointer to signal handler

    ///////////////////////////////////////////////////////
    // avoid large motion at starting points 
    ///////////////////////////////////////////////////////
    for(int index=0; index<10; index=index+1)    
    {   
        pos_1 = motor_1.set_torque(motor_id_1, torque_1, &d_theta_1_t, &torque_1_t);   
        pos_2 = motor_2.set_torque(motor_id_2, torque_2, &d_theta_2_t, &torque_2_t);   
    }   

    ///////////////////////////////////////////////////////
    // tracking all way points 
    ///////////////////////////////////////////////////////
    for (int epi=0; epi < num_episodes; epi=epi+1)    
    {    
        // Catch a Ctrl-C event:    
        signal(SIGINT, sig_h);   

        for (int index = 0; index<Num_waypoints; index=index+1)
        { 
            /////////////////////////////////////////////////////
            // read-time data  
            ///////////////////////////////////////////////////// 
            theta_1_t = motor_1.read_sensor(motor_id_1) - theta_1_initial;    
            theta_2_t = -1 * (motor_2.read_sensor(motor_id_2) + theta_1_t - theta_2_initial);    

            theta_t_list[0] = theta_1_t;    
            theta_t_list[1] = theta_2_t;    

            theta_e_list[0] = theta_1_list[index];    
            theta_e_list[1] = theta_2_list[index];    

            d_theta_t_list[0] = d_theta_1_t;    
            d_theta_t_list[1] = d_theta_2_t;    

            params[0] = stiff_1_list[index];     
            params[1] = stiff_2_list[index];     
            params[2] = damping_1_list[index];     
            params[3] = damping_2_list[index];     

            /////////////////////////////////////////////////////
            // set torque control command 
            ///////////////////////////////////////////////////// 
            /// Joint space control
            calculate_joint_torque(
            params,   
            theta_e_list, d_theta_e_list,   
            theta_t_list, d_theta_t_list,   
            torque_1, torque_2   
            );   

            // /// Task space control
            // calculate_task_torque( 
            // params,  
            // theta_e_list, d_theta_e_list,  
            // theta_t_list, d_theta_t_list,  
            // torque_1, torque_2  
            // );   

            torque_1_o = torque_1/ctl_ratio_1;     
            torque_2_o = torque_2/ctl_ratio_2;    

            pos_1 = motor_1.set_torque(motor_id_1, torque_1, &d_theta_1_t, &torque_1_t);     
            pos_2 = motor_2.set_torque(motor_id_2, torque_2, &d_theta_2_t, &torque_2_t);    


            d_theta_1_t = filter(d_theta_1_old, d_theta_1_t);   
            d_theta_2_t = filter(d_theta_2_old, d_theta_1_t);   

            d_theta_1_old = d_theta_1_t;    
            d_theta_2_old = d_theta_2_t;    

            ////////////////////////////////////////////////////////
            // Save Data
            ////////////////////////////////////////////////////////
            OutFileAngle << theta_e_list[0] << "," << theta_t_list[0] << "," << d_theta_t_list[0] << "," << theta_e_list[1] << "," << theta_t_list[1] << "," << d_theta_t_list[1] << "\n";   

            OutFileTorque << torque_1_o << "," << torque_1_t << "," << torque_2_o << "," << torque_2_t << "\n";  
            
            if (run_on==0) 
            {
                break; 
            } 
            else
            { 

            } 
        }
        if (run_on==0) 
        {
            break; 
        } 
        else
        {

        }
    }    
    
    OutFileTorque.close();    
    OutFileAngle.close();    
    motor_1.pack_stop_cmd(motor_id_1);      
    motor_2.pack_stop_cmd(motor_id_2);       

    return 1;   
}   


namespace py = pybind11; 

PYBIND11_MODULE(motor_control, m) {
    m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------

        .. currentmodule:: cmake_example

        .. autosummary::
           :toctree: _generate

           add
           subtract
    )pbdoc"; 

    m.def(
        "Jacobian", &Jacobian, R"pbdoc( 
        Calculate Jacobian

        Some other explanation about the add function. 
    )pbdoc"); 

    m.def(
        "Cal_torque", &Cal_torque, R"pbdoc( 
        Calculate torque

        Some other explanation about the add function. 
    )pbdoc"); 

    m.def(
        "load_path_data", &load_path_data, R"pbdoc( 
        load path data from .txt file

        Some other explanation about the add function. 
    )pbdoc"); 

    m.def(
        "read_angle_1", &read_angle_1, R"pbdoc( 
        read angle 1

        Some other explanation about the add function. 
    )pbdoc"); 

    m.def(
        "read_angle_2", &read_angle_2, R"pbdoc( 
        read angle 2

        Some other explanation about the add function. 
    )pbdoc"); 

    m.def(
        "read_angle_3", &read_angle_3, R"pbdoc( 
        read angle 3

        Some other explanation about the add function. 
    )pbdoc"); 

    m.def(
        "read_analog_encoder", &read_analog_encoder, R"pbdoc( 
        read_analog_encoder

        Some other explanation about the add function. 
    )pbdoc"); 

    // m.def(
    //     "read_initial_encode", &read_initial_encode, R"pbdoc( 
    //     read_initial_encode

    //     Some other explanation about the add function. 
    // )pbdoc"); 

    m.def(
        "read_encoder_angles", &read_encoder_angles, R"pbdoc( 
        read_encoder_angles

        Some other explanation about the add function. 
    )pbdoc"); 

    m.def(
        "set_position", &set_position, R"pbdoc( 
        set_position

        Some other explanation about the add function. 
    )pbdoc"); 

    m.def(
        "set_two_link_position", &set_two_link_position, R"pbdoc( 
        set_two_link_position

        Some other explanation about the add function. 
    )pbdoc"); 

    m.def(
        "Convert_stiffness", &Convert_stiffness, R"pbdoc( 
        Convert_stiffness  

        Some other explanation about the add function. 
    )pbdoc");  

    m.def(
        "motor_3_stop", &motor_3_stop, R"pbdoc( 
        motor_3_stop

        Some other explanation about the add function. 
    )pbdoc");  

    m.def(
        "motor_two_link_stop", &motor_two_link_stop, R"pbdoc( 
        motor_two_link_stop

        Some other explanation about the add function. 
    )pbdoc"); 

    m.def(
        "read_initial_angle_1", &read_initial_angle_1, R"pbdoc( 
        read_initial_angle_1 

        Some other explanation about the add function.  
    )pbdoc"); 

    m.def(
        "read_initial_angle_2", &read_initial_angle_2, R"pbdoc( 
        read_initial_angle_2

        Some other explanation about the add function. 
    )pbdoc"); 

    m.def(
        "read_initial_angle_3", &read_initial_angle_3, R"pbdoc( 
        read_initial_angle_3

        Some other explanation about the add function. 
    )pbdoc"); 

    m.def("get_demonstration", &get_demonstration, R"pbdoc(
        get_demonstration

        Some other explanation about the add function.
    )pbdoc");  

    // m.def("phri_get_demonstration", &phri_get_demonstration, R"pbdoc(
    //     phri_get_demonstration

    //     Some other explanation about the add function.
    // )pbdoc"); 

    m.def("run_one_loop", &run_one_loop, R"pbdoc( 
        run_one_loop 

        Some other explanation about the add function. 
    )pbdoc"); 

    m.def("move_to_target_point", &move_to_target_point, R"pbdoc(
        move_to_target_point

        Some other explanation about the add function.
    )pbdoc"); 
    
    // run_one_loop(double stiffness, double damping, double theta_1_initial, double theta_2_initial, int Num_waypoints)


#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev"; 
#endif
}