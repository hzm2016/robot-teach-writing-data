using namespace std; 
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <stdlib.h>
#include <pybind11/pybind11.h>
#include <GyemsCan/gyems_can_functions.h>

void split(const string& s, vector<string>& tokens, const string& delim=",") 
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

int add(int i, int j) {
    return i + 2 * j;
}

// int run_one_loop()
// {
//     ////////////////////////////////////////////////////////
//     // Initial hardware ::: can device
//     ////////////////////////////////////////////////////////

//     std::cout << "Initial Can0 and Can1 !!!" << std::endl;  
//     CANDevice can0((char *) "can0");  
//     can0.begin(); 
//     CANDevice can1((char *) "can1");  
//     can1.begin(); 

//     Gcan motor_1(can1); 
//     Gcan motor_2(can0);   
//     motor_1.begin(); 
//     motor_2.begin(); 

//     std::cout << "Enable Motors Done!!!" << std::endl; 
    
//     ////////////////////////////////////////////////////////
//     // Load path from txt file
//     ////////////////////////////////////////////////////////
//     int Num_waypoints = 19999; 
//     int index_point = 0;  
//     double d_t = 0.001; 

//     double theta_1_list[Num_waypoints]; 
//     double theta_2_list[Num_waypoints]; 

//     string input_path = "2_font_3_angle_list.txt"; 
//     ifstream input_angle_list; 
//     input_angle_list.open(input_path); 
//     string s; 
//     vector<string> angle_list; 

//     while(getline(input_angle_list, s))
//     {
//         // cout << s << endl; 
//         split(s, angle_list, ","); 
//         theta_1_list[index_point] = atof(angle_list[0].c_str()); 
//         // printf("theta 1: %f\n", theta_1_list[index_point]); 
//         theta_2_list[index_point] = atof(angle_list[1].c_str()); 
//         // printf("theta 2: %f\n", theta_2_list[index_point]); 

//         index_point += 1; 
//     }
//     printf("Num_waypoints: %d\n", index_point);  
//     input_angle_list.close(); 

//     ////////////////////////////////////////////////////////
//     // Define file to store data
//     ////////////////////////////////////////////////////////

//     // string root_path = "home/zhimin/code/8_nus/robotic-teaching/control/motor_control"; 
//     string root_path = "data/"; 

//     string output_torque = root_path + "torque_list.txt"; 
//     ofstream OutFileTorque(output_torque); 
//     OutFileTorque << "torque_1" << " " << "torque_2" << "\n";  

//     // string output_torque_1 = root_path + "torque_list_1.txt"; 
//     // ofstream OutFileTorque1; 
//     // OutFileTorque1.open(output_torque_1.c_str()); 

//     string output_angle = root_path + "angle_list.txt"; 
//     ofstream OutFileAngle(output_angle); 
//     OutFileAngle << "angle_1" << " " << "angle_2" << "\n";  

//     // string output_angle_2 = root_path + "angle_list_2.txt"; 
//     // ofstream OutFileAngle2(output_angle_2); 

//     string output_vel = root_path + "angle_vel_list.txt"; 
//     ofstream OutFileVel(output_vel); 
//     OutFileVel << "vel_1" << " " << "vel_2" << "\n"; 

//     // string output_vel_2 = root_path + "angle_vel_list_2.txt"; 
//     // ofstream OutFileVel2(output_vel_2); 

//     ////////////////////////////////////////////////////////
//     // Initial Calibration
//     ////////////////////////////////////////////////////////
//     // double theta_1_initial = motor_1.read_sensor(2); 
//     // printf(" Motor 1 initial position: %f\n", theta_1_initial); 

//     // double theta_2_initial = motor_2.read_sensor(1); 
//     // printf(" Motor 2 initial position: %f\n", theta_2_initial); 

//     double theta_1_initial = -0.294288; 
//     double theta_2_initial = 0.402938; 

//     ////////////////////////////////////////////////////////
//     // Impedance Parameters
//     ////////////////////////////////////////////////////////
// 	double K_p_1 = 16;  
// 	double K_d_1 = 0.8; 
//     double K_p_2 = 14;  
// 	double K_d_2 = 0.8; 
// 	double K_i = 0.0; 

//     double torque_lower_bound = -2.5;  
//     double torque_upper_bound = 2.5;  
    
//     // depends on the relation of torque and current
//     double ctl_ratio_1 = - 2000.0/32;  
//     double ctl_ratio_2 = 2000.0/32;  

//     double theta_1_t = 0.0; 
//     double theta_2_t = 0.0; 

//     double d_theta_1_t = 0.0;  
//     double d_theta_2_t = 0.0;  

//     double theta_1_e = 0.0;  
//     double theta_2_e = 0.0;  

//     double d_theta_1_e = 0.0;  
//     double d_theta_2_e = 0.0;  

//     double torque_1 = 0.0;  
//     double torque_2 = 0.0;  

//     double torque_1_t = 0.0;  
//     double torque_2_t = 0.0;  

//     double pos_1 = 0.0;  
//     double pos_2 = 0.0;  

//     ////////////////////////////////////////////////////////
//     // One loop control
//     ////////////////////////////////////////////////////////

//     // while(true) 
//     // {
//     //     theta_1_t = motor_1.read_sensor(2) - theta_1_initial;  
//     //     theta_2_t = -1 * (motor_2.read_sensor(1) + theta_1_t - theta_2_initial);  

//     //     printf(" theta_1_t: %f\n", theta_1_t);  
//     //     printf(" theta_2_t: %f\n", theta_2_t);  

//     //     OutFileAngle << theta_1_t << " " << theta_2_t << "\n"; 

//     //     // torque_1 = clip(-1 * K_p * (theta_1_e - theta_1_t) - K_d * (d_theta_1_e - d_theta_1_t), torque_lower_bound, torque_upper_bound);  
//     //     // torque_2 = clip(-1 * K_p * (theta_2_e - theta_2_t) - K_d * (d_theta_2_e - d_theta_2_t), torque_lower_bound, torque_upper_bound); 

//     //     pos_1 = motor_1.set_torque(2, 0, &d_theta_1_t, &torque_1_t);  
//     //     pos_2 = motor_2.set_torque(1, 0, &d_theta_2_t, &torque_2_t);  

//     //     OutFileTorque << torque_1_t << " " << torque_2_t << "\n"; 

//     //     OutFileVel << d_theta_1_t << " " << d_theta_2_t << "\n"; 

//     //     printf("d_theta_1_t: %f\n", d_theta_1_t); 
//     //     printf("d_theta_2_t: %f\n", d_theta_2_t); 
//     // }
    

//     // for(int index=0; index<Num_waypoints; index=index+1) 
//     // {
//     //     theta_1_e = theta_1_list[index]; 
//     //     theta_2_e = theta_2_list[index]; 

//     //     cout << "index" << index << "\n";
//     //     cout << "theta_1_e" << theta_1_e << "\n";
//     //     cout << "theta_2_e" << theta_2_e << "\n";
//     // }
//     for(int index=0; index<5; index=index+1) 
//     {
//         pos_1 = motor_1.set_torque(2, 0.0, &d_theta_1_t, &torque_1_t); 
//         pos_2 = motor_2.set_torque(1, 0.0, &d_theta_2_t, &torque_2_t); 
//     }

//     for(int index=0; index<Num_waypoints; index=index+1) 
//     {
//         theta_1_e = theta_1_list[index]; 
//         theta_2_e = theta_2_list[index]; 

//         if(index==0) 
//         {
//             d_theta_1_e = 0.0; 
//             d_theta_2_e = 0.0; 
//         }
//         else 
//         {
//             d_theta_1_e = (theta_1_list[index] - theta_1_list[index-1])/d_t; 
//             d_theta_2_e = (theta_2_list[index] - theta_2_list[index-1])/d_t; 
//         }
        
//         theta_1_t = motor_1.read_sensor(2) - theta_1_initial;  
//         theta_2_t = -1 * (motor_2.read_sensor(1) + theta_1_t - theta_2_initial);  

//         // set torque control command 
//         torque_1 = clip(- K_p_1 * (theta_1_e - theta_1_t) - K_d_1 * (d_theta_1_e - d_theta_1_t), torque_lower_bound, torque_upper_bound) * ctl_ratio_1; 
//         torque_2 = clip(- K_p_2 * (theta_2_e - theta_2_t) - K_d_2 * (d_theta_2_e - d_theta_2_t), torque_lower_bound, torque_upper_bound) * ctl_ratio_2; 

//         double torque_1_o = - K_p_1 * (theta_1_e - theta_1_t) - K_d_1 * (d_theta_1_e - d_theta_1_t); 
//         double torque_2_o = - K_p_2 * (theta_2_e - theta_2_t) - K_d_2 * (d_theta_2_e - d_theta_2_t); 

//         // printf("input_torque_1_t: %f\n", torque_1); 
//         // printf("input_torque_2_t: %f\n", torque_1); 

//         pos_1 = motor_1.set_torque(2, torque_1, &d_theta_1_t, &torque_1_t); 
//         pos_2 = motor_2.set_torque(1, torque_2, &d_theta_2_t, &torque_2_t); 
//         // pos_2 = motor_2.set_torque(1, torque_2, &d_theta_2_t, &torque_2_t); 

//         ////////////////////////////////////////////////////////
//         // Save Data
//         ////////////////////////////////////////////////////////

//         OutFileAngle << theta_1_t << " " << theta_2_t << "\n";  

//         OutFileTorque << torque_1_o << " " << torque_2_o << "\n"; 

//         OutFileVel << d_theta_1_t << " " << d_theta_2_t << "\n"; 
//     }

//     OutFileTorque.close(); 
//     OutFileAngle.close(); 
//     OutFileVel.close(); 

//     motor_1.pack_stop_cmd(2); 
//     motor_2.pack_stop_cmd(1);  

//     return 0; 
// }

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

    m.def("add", &add, R"pbdoc(
        Add two numbers

        Some other explanation about the add function.
    )pbdoc");

    m.def("subtract", [](int i, int j) { return i - j; }, R"pbdoc(
        Subtract two numbers

        Some other explanation about the subtract function.
    )pbdoc");

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}