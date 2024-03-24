#include <iostream>  
#include <fstream>  
#include <string>  
#include <vector>  
#include <stdlib.h>  

using namespace std; 

#include <unistd.h>   
#include <signal.h>   
#include <cmath>   
#include <stdio.h>   

#include <Eigen/Dense>  

using namespace Eigen;  

#include "gyems_can_functions.h" 
#include "renishaw_can_functions.hpp" 
#include "robot_cal.hpp"


int read_initial_encode(double encoder_angle[2]) 
{ 
    ////////////////////////////////////////////
    // Read original encoder
    ////////////////////////////////////////////
    controller_renishaw encoder("can2");  

    float encoder_arr[2];  

	encoder.read_ang_encoder(encoder_arr);  
     
  	encoder_angle[0] = (double) encoder_arr[1]*PI/180.0;   
  	encoder_angle[1] = (double) encoder_arr[0]*PI/180.0;   
    
    // printf("Encoder 1 position: %f\n", encoder_angle[0]);   
    // printf("Encoder 2 position: %f\n", encoder_angle[1]);    
    return 1;  
}  


double read_link_angle_1(double q_1_initial)   
{
    ////////////////////////////////////////////  
    // Read link angle 1
    ////////////////////////////////////////////  

    controller_renishaw encoder("can2");  

    float encoder_arr[2];  

	encoder.read_ang_encoder(encoder_arr);  

  	double q_1 = (double) encoder_arr[1]*PI/180.0 + q_1_initial;  

    return q_1;  
} 


double read_link_angle_2(double q_2_initial)   
{
    ////////////////////////////////////////////   
    // Read link angle 2    
    ////////////////////////////////////////////   

    controller_renishaw encoder("can2");  

    float encoder_arr[2];  

	encoder.read_ang_encoder(encoder_arr);  

  	double q_2 = (double) encoder_arr[0]*PI/180.0 + q_2_initial;  

    return q_2;  
}  


double read_initial_angle_1() 
{
    ////////////////////////////////////////////
    // Read motor original angle 1
    ////////////////////////////////////////////

    CANDevice can1((char *) "can1");   
    can1.begin();   

    Gcan motor_1(can1);   
    motor_1.begin();   
    
    double theta_1 = motor_1.read_sensor(motor_id_1);   
    printf("Motor 1 original position: %f\n", theta_1);   

    return theta_1;   
}


double read_initial_angle_2()   
{
    ////////////////////////////////////////////
    // Read motor original angle 2
    ////////////////////////////////////////////

    CANDevice can0((char *) "can0");   
    can0.begin();   

    Gcan motor_2(can0);   
    motor_2.begin();   

    // double theta_2 = motor_2.read_sensor(1);   
    double theta_2 = motor_2.read_sensor(motor_id_2);  
    printf("Motor 2 original position: %f\n", theta_2);   

    return theta_2;   
}


double read_initial_angle_3()   
{
    ////////////////////////////////////////////
    // Read motor original angle 3
    ////////////////////////////////////////////
    CANDevice can3((char *) "can2");   
    can3.begin();   

    Gcan motor_3(can3);   
    motor_3.begin();   

    double theta_3 = motor_3.read_single_turn(motor_id_3);  
    printf("Motor 3 original position: %f\n", theta_3);   

    return theta_3;   
}


double read_angle_1(double theta_1_initial) 
{
    //////////////////////////////////////////// 
    // Read motor angle 1
    //////////////////////////////////////////// 

    CANDevice can1((char *) "can1");   
    can1.begin();   

    Gcan motor_1(can1);   
    motor_1.begin();   
    
    double theta_1 = motor_1.read_sensor(motor_id_1) - theta_1_initial;    
    printf("Motor 1 position: %f\n", theta_1);   

    return theta_1;   
}


double read_angle_2(double theta_2_initial, double theta_1_t)   
{
    ////////////////////////////////////////////
    // Read motor angle 2
    ////////////////////////////////////////////

    CANDevice can0((char *) "can0");    
    can0.begin();   

    Gcan motor_2(can0);   
    motor_2.begin();   

    double theta_2 = -1 * (motor_2.read_sensor(1) + theta_1_t - theta_2_initial);  
    // double theta_2 = motor_2.read_sensor(motor_id_2) - theta_2_initial;   
    printf("Motor 2 position: %f\n", theta_2);  

    return theta_2;   
}


double read_angle_3(double theta_3_initial)    
{
    ////////////////////////////////////////////
    // Read motor angle 3 
    ////////////////////////////////////////////
    CANDevice can3((char *) "can2");   
    can3.begin();   

    Gcan motor_3(can3);   
    motor_3.begin();   

    // double theta_2 = -1 * (motor_2.read_sensor(1) + theta_1_t - theta_2_initial);   
    // printf("Motor 2 position: %f\n", theta_2);   
    // double theta_3 = motor_3.read_sensor(1);  
    // printf("Motor 3 position: %f\n", theta_3);   

    double theta_3_t; 

    // theta_3_t = -1 * (motor_3.read_sensor(1));   
    // printf("Motor 3 position: %f\n", theta_3_t);   

    theta_3_t = motor_3.read_single_turn(motor_id_3) - theta_3_initial;   
    printf("Motor 3 rad: %f, deg %f\n", theta_3_t, theta_3_t * 180/3.14);   

    return theta_3_t;   
}


int vic_optimization( 
double stiffness, double damping,  
double q_1_target, double q_2_target,  
double q_1_initial, double q_2_initial,  
double theta_1_initial, double theta_2_initial,  
double dist_threshold   
) 
{
    ////////////////////////////////////////////////////////
    //// Initial Encoder and SEA Motor CAN
    //////////////////////////////////////////////////////// 

    CANDevice can0((char *) "can0");    
    can0.begin();   
    CANDevice can1((char *) "can1");   
    can1.begin();   

    Gcan motor_1(can1);   
    Gcan motor_2(can0);   
    motor_1.begin();   
    motor_2.begin();   

    printf("Move to target start !!!!\n");  

    ///////////////////// encoder reading //////////////////
    controller_renishaw encoder("can2");  
    float encoder_arr[2];  

    ////////////////////////////////////////////////////////
    // One loop control demonstration   
    ////////////////////////////////////////////////////////

    string output_angle = "move_target_angle_list.txt";    
    ofstream OutFileAngle(output_angle);    
    OutFileAngle << "angle_1" << "," << "angle_2" << "\n";    

    string output_torque = "move_target_torque_list.txt";    
    ofstream OutFileTorque(output_torque);    
    OutFileTorque << "torque_1" << "," << "torque_2" << "\n";    

    double torque_lower_bound = -1.5;    
    double torque_upper_bound = 1.5;   
    
    double ctl_ratio_1 = -2000.0/32;   
    double ctl_ratio_2 = 2000.0/32;   

    double theta_1_t = 0.0;   
    double theta_2_t = 0.0;   

    double d_theta_1_t = 0.0;    
    double d_theta_2_t = 0.0;    

    double theta_1_e = 0.0;   
    double theta_2_e = 0.0;   

    double d_theta_1_e = 0.0;   
    double d_theta_2_e = 0.0;   

    double torque_1 = 0.0;   
    double torque_2 = 0.0;   

    double torque_1_t = 0.0;  
    double torque_2_t = 0.0;  

    double pos_1 = 0.0;  
    double pos_2 = 0.0;      

    double q_1_t = 0.0; 
    double q_2_t = 0.0; 

    double dist = 0.0; 
    int initial_index = 0; 
    int max_index = 10000; 

    /////////////////////////////////////////////////////
    /////  avoid large motion at starting points  ///////
    /////////////////////////////////////////////////////
    for(int index=0; index<5; index=index+1) 
    {
        pos_1 = motor_1.set_torque(2, 0.0, &d_theta_1_t, &torque_1_t); 
        pos_2 = motor_2.set_torque(1, 0.0, &d_theta_2_t, &torque_2_t); 
    }

    // run_on = 1; 

    // // Catch a Ctrl-C event:
	// void  (*sig_h)(int) = sigint_1_step;   // pointer to signal handler

    // // Catch a Ctrl-C event: 
    // signal(SIGINT, sig_h);  
 
    int run_on = 1;

    // dist > dist_threshold && initial_index < max_index
    while(run_on)  
    {
        ////////////////////////////////////////////////////////
        //// Initial Encoder and SEA Motor CAN
        ////////////////////////////////////////////////////////
        theta_1_t = motor_1.read_sensor(2) - theta_1_initial;  
        theta_2_t = -1 * (motor_2.read_sensor(1) + theta_1_t - theta_2_initial);  

        encoder.read_ang_encoder(encoder_arr);  

        q_1_t = (double) encoder_arr[1]*PI/180.0 - q_1_initial;  
        q_2_t = (double) encoder_arr[0]*PI/180.0 - q_2_initial;  

        dist = sqrt(pow((theta_1_t - q_1_target), 2) + pow((theta_2_t - q_2_target), 2));   

        printf(" theta_1_t: %f\n", theta_1_t);    
        printf(" theta_2_t: %f\n", theta_2_t);    

        /////////////////////////////////////////////////////
        //// calculate torque control command 
        ///////////////////////////////////////////////////// 
        torque_1 = clip(- stiffness * (q_1_target - theta_1_t) - damping * (d_theta_1_e - d_theta_1_t), torque_lower_bound, torque_upper_bound) * ctl_ratio_1; 
        torque_2 = clip(- stiffness * (q_2_target - theta_2_t) - damping * (d_theta_2_e - d_theta_2_t), torque_lower_bound, torque_upper_bound) * ctl_ratio_2; 

        // double torque_1_o = - K_p_1 * (theta_1_e - theta_1_t) - K_d_1 * (d_theta_1_e - d_theta_1_t);  
        // double torque_2_o = - K_p_2 * (theta_2_e - theta_2_t) - K_d_2 * (d_theta_2_e - d_theta_2_t);  

        OutFileAngle << theta_1_t << "," << theta_2_t << "\n";   

        // pos_1 = motor_1.set_torque(2, torque_1, &d_theta_1_t, &torque_1_t);    
        // pos_2 = motor_2.set_torque(1, torque_2, &d_theta_2_t, &torque_2_t);    

        pos_1 = motor_1.set_torque(2, 0.0, &d_theta_1_t, &torque_1_t);   
        pos_2 = motor_2.set_torque(1, torque_2, &d_theta_2_t, &torque_2_t);   

        OutFileTorque << torque_1_t << "," << torque_2_t << "\n";   

        // OutFileVel << d_theta_1_t << " " << d_theta_2_t << "\n";   

        // printf("d_theta_1_t: %f\n", d_theta_1_t);   
        // printf("d_theta_2_t: %f\n", d_theta_2_t);   
    }

    printf("Move to target done !!!! \n"); 

    OutFileAngle.close();   
    OutFileTorque.close();       

    motor_1.pack_stop_cmd(2);   
    motor_2.pack_stop_cmd(1);    

    return 1;  
}


void load_path_data(double *theta_1_list, double *theta_2_list)
{
    ////////////////////////////////////////////////////////
    // Load path from txt file
    ////////////////////////////////////////////////////////   
    int index_point = 0;    

    // int Num_waypoints = 19999;  
    // double theta_1_list[Num_waypoints];    
    // double theta_2_list[Num_waypoints];     

    string input_path = "angle_list.txt";    
    ifstream input_angle_list;    
    input_angle_list.open(input_path);    
    string s;  
    vector<string> angle_list;    

    while(getline(input_angle_list, s))  
    {
        // cout << s << endl;  
        split(s, angle_list, ",");   
        theta_1_list[index_point] = atof(angle_list[0].c_str());   
        // printf("theta 1: %f\n", theta_1_list[index_point]);  

        theta_2_list[index_point] = atof(angle_list[1].c_str());   
        // printf("theta 2: %f\n", theta_2_list[index_point]);  

        index_point += 1;  
    }
    printf("Num_waypoints: %d\n", index_point);   
    input_angle_list.close();   
} 


int motor_two_link_stop()
{
    ////////////////////////////////////////////
    // Read motor angle 3 
    ////////////////////////////////////////////
    CANDevice can1((char *) "can1");  
    can1.begin();  

    Gcan motor_1(can1);   
    motor_1.begin();  
    
    double theta_1 = motor_1.read_sensor(2);  

    CANDevice can0((char *) "can0");   
    can0.begin();   

    Gcan motor_2(can0);   
    motor_2.begin();   

    double theta_2 = -1 * (motor_2.read_sensor(1) + theta_1);   

    motor_1.pack_stop_cmd(2);  
    motor_2.pack_stop_cmd(2); 
    printf("First two motors stop !!!\n");    
    
    return 1; 
}


int motor_3_stop()
{
    ////////////////////////////////////////////
    // Read motor angle 3 
    ////////////////////////////////////////////
    CANDevice can0((char *) "can2");   
    can0.begin();   

    Gcan motor_3(can0);   
    motor_3.begin();   
 
    double theta_3_t;   

    theta_3_t = motor_3.read_single_turn(1);  

    motor_3.pack_stop_cmd(1);  
    printf("Motor stop !!! and final pos: %f\n", theta_3_t);   
    
    return 1; 
}


double clip(double angle, double lower_bound, double upper_bound)
{
    double clip_angle; 

    if (angle < lower_bound)
    {
        clip_angle = lower_bound; 
    } 
    else if(angle > upper_bound)
    {
        clip_angle = upper_bound; 
    }
    else
    {
        clip_angle = angle; 
    }
    return clip_angle; 
}


void split(const string& s, vector<string>& tokens, const string& delim)
{
    tokens.clear(); 
    size_t lastPos = s.find_first_not_of(delim, 0); 
    size_t pos = s.find(delim, lastPos); 
    while (lastPos != string::npos) {
        tokens.emplace_back(s.substr(lastPos, pos - lastPos)); 
        lastPos = s.find_first_not_of(delim, pos); 
        pos = s.find(delim, lastPos); 
    }  
}  


MatrixXd Jacobian(double theta_1_t, double theta_2_t) 
{
    // const <MatrixXd> J 
    MatrixXd m(2, 2); 

    m(0, 0) = - L_1 * sin(theta_1_t) - L_2 * sin(theta_1_t + theta_2_t);  
    m(0, 1) = - L_2 * sin(theta_1_t + theta_2_t);  
    m(1, 0) = L_1 * cos(theta_1_t) + L_2 * cos(theta_1_t + theta_2_t);   
    m(1, 1) = L_2 * cos(theta_1_t + theta_2_t);  

    // printf("matrix :%f\n", clip(m(1, 1), -1, 1)); 

    return m; 
}

Vector2d Forward_ik(double theta_1_t, double theta_2_t)
{
    /// forward kinematics 
    Vector2d pos_t(0.0, 0.0);  
    
    pos_t(0) = L_1 * cos(theta_1_t) + L_2 * cos(theta_1_t + theta_2_t);   
    pos_t(1) = L_1 * sin(theta_1_t) + L_2 * sin(theta_1_t + theta_2_t);   

    return pos_t;   
}


double
filter(double d_angel_old, double d_angle_new) {
    double weight_filter = 0.95;
    return weight_filter * d_angel_old + (1 - weight_filter) * d_angle_new;
}

int read_encoder_angles(double q_1_initial, double q_2_initial) 
{
    double encoder_values[2] = {0.0, 0.0};  
    read_initial_encode(encoder_values);  
    printf("encoder1: %f, encoder2: %f", encoder_values[0] - q_1_initial, encoder_values[1] - q_2_initial); 

    return 1;  
}


double read_analog_encoder()
{
    ///////////////////////////////////////////////////////////////////////////
	// Initialize Sensoray 526:
	///////////////////////////////////////////////////////////////////////////

    const int NUM_ADC_CHAN			= 6; 
    
    int32_t ADC_CHAN[NUM_ADC_CHAN]	= {0, 1, 2, 3, 4, 5};  
    // int32_t ADC_CHAN[NUM_ADC_CHAN]	= {7, 6, 5, 4, 3, 2, 1, 0}; 

    double adc_data[NUM_ADC_CHAN]	= {0, 0, 0, 0, 0, 0};  

    cout << "initial s526 !!!" << endl;  

    // s526_read_id();  

	// // Initialize hardware: 
	// s526_init();  

    // cout << "initial DAC !!!" << endl; 
    
    // s526_adc_init(ADC_CHAN, NUM_ADC_CHAN);  

    // cout << "Test ADC read !!!" << endl; 

    // // Read ADC:
    // s526_adc_read(ADC_CHAN, NUM_ADC_CHAN, adc_data); 

    // printf("FT data:: Tz %f\t Ty: %f\t Tx: %f Fz %f\t Fy: %f\t Fx: %f\n", adc_data[0], adc_data[1], adc_data[2], adc_data[3], adc_data[4], adc_data[5]);

} 
