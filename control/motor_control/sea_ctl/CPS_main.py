int control_single_motor(
double stiffness_1, double stiffness_2,  
double damping_1, double damping_2,  
py::array_t<double> q_1_target, py::array_t<double> q_2_target, int N,  
double theta_1_initial, double theta_2_initial,  
double dist_threshold   
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

    // printf("Move to target point start !!!!\n");   

    ////////////////////////////////////////////////////////
    // One loop control demonstration
    ////////////////////////////////////////////////////////
    string output_angle = "move_target_angle_list.txt";     
    ofstream OutFileAngle(output_angle);    
    OutFileAngle << "angle_1" << "," << "angle_2" << "," << "torque_1" << "," << "torque_t" << "," << "d_theta_t" << "\n";     

    string output_torque = "move_target_torque_list.txt";    
    ofstream OutFileTorque(output_torque);   
    OutFileTorque << "torque_1" << "," << "torque_2" << "," << "torque_2" << "," << "torque_t" << "," << "d_theta_t" << "\n";    

    double torque_lower_bound = -2.5;    
    double torque_upper_bound = 2.5;   

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

    double torque_1_o = 0.0; 
    double torque_2_o = 0.0; 

    double pos_1 = 0.0;      
    double pos_2 = 0.0;      
 
    double dist = 1.0;   

    // double ratio_ctl_1 = -2000/32; 
    // double ratio_ctl_2 = -2000/32; 

    py::buffer_info q_1_list_buf = q_1_target.request();    
    py::buffer_info q_2_list_buf = q_2_target.request();    
    double *q_1_list = (double *)q_1_list_buf.ptr; 
    double *q_2_list = (double *)q_2_list_buf.ptr;  

    run_on = 1;  

    // Catch a Ctrl-C event:
	void  (*sig_h)(int) = sigint_1_step;   // pointer to signal handler
 
    int index = 0; 

    // dist > dist_threshold && initial_index < max_index
    while(run_on && index<N)  
    {
        // Catch a Ctrl-C event: 
        signal(SIGINT, sig_h);  

        theta_1_t = motor_1.read_sensor(2) - theta_1_initial;    
        theta_2_t = motor_2.read_sensor(1) - theta_2_initial;    

        dist = sqrt(pow((theta_1_t - q_1_list[index]), 2) + pow((theta_2_t - q_2_list[index]), 2));    
        // printf("theta_1_t: %f\n", theta_1_t);     
        // printf("theta_2_t: %f\n", theta_2_t);     

        /////////////////////////////////////////////////////  
        // calculate torque control command    
        ///////////////////////////////////////////////////// 
        torque_1 = clip(-1 * stiffness_1 * (q_1_list[index] - theta_1_t) - damping_1 * (d_theta_1_e - d_theta_1_t), torque_lower_bound, torque_upper_bound) * ctl_ratio_1; 
        torque_2 = clip(-1 * stiffness_2 * (q_2_list[index] - theta_2_t) - damping_2 * (d_theta_2_e - d_theta_2_t), torque_lower_bound, torque_upper_bound) * ctl_ratio_2; 

        torque_1_o = clip(-1 * stiffness_1 * (q_1_list[index] - theta_1_t) - damping_1 * (d_theta_1_e - d_theta_1_t), torque_lower_bound, torque_upper_bound);  
        torque_2_o = clip(-1 * stiffness_2 * (q_2_list[index] - theta_2_t) - damping_2 * (d_theta_2_e - d_theta_2_t), torque_lower_bound, torque_upper_bound);  

        // OutFileAngle << theta_1_t << "," << theta_2_t << "\n";    
        OutFileAngle << q_1_list[index] << "," << theta_1_t << "," << torque_1_o << "," << torque_1_t << "," << d_theta_1_t <<"\n";    

        pos_1 = motor_1.set_torque(2, torque_1, &d_theta_1_t, &torque_1_t);     
        pos_2 = motor_2.set_torque(1, torque_2, &d_theta_2_t, &torque_2_t);       

        // pos_1 = motor_1.set_torque(2, torque_1, &d_theta_1_t, &torque_1_t);   
        // pos_2 = motor_2.set_torque(1, torque_2, &d_theta_2_t, &torque_2_t);   

        OutFileTorque << q_2_list[index] << "," << theta_2_t << "," << torque_2_o << "," << torque_2_t << "," << d_theta_2_t <<"\n";   

        // OutFileVel << d_theta_1_t << " " << d_theta_2_t << "\n";   

        // printf("d_theta_1_t: %f\n", d_theta_1_t);   
        // printf("d_theta_2_t: %f\n", d_theta_2_t);  
        index = index + 1;   
    } 


    // printf("dist : %f\n", dist);  

    // printf("dist : %f\n", dist); 

    OutFileAngle.close();   
    OutFileTorque.close();       

    motor_1.pack_stop_cmd(2);   
    motor_2.pack_stop_cmd(1);   

    // printf("Move to target point done !!!! \n");   

    return 1;   
}


int rotate_to_target(  
    double stiffness, double damping,  
    double theta_target,  
    double theta_initial,   
    double dist_threshold,    
    int32_t torque_cmd
)
{
    ////////////////////////////////////////////////////////
    //// Initial Encoder and Motor CAN
    //////////////////////////////////////////////////////// 
    CANDevice can3((char *) "can3");   
    can3.begin();   

    Gcan motor_3(can3);   
    motor_3.begin();  

    printf("Rotate to target point !!!!\n");   

    double torque_lower_bound = -1.0;      
    double torque_upper_bound = 1.0;      

    double theta_t = 0.0;   

    double d_theta_t = 0.0;    

    double theta_e = 0.0;  

    double d_theta_e = 0.0;   

    double torque = 0.0;  

    double torque_t = 0.0;   

    double pos_1 = 0.0;  
    double curr_t = 0.0;        

    double dist = 0.0;  

    string output_angle = "real_angle_list.txt";   
    ofstream OutFileAngle(output_angle);   
    OutFileAngle << "angle_1" << "," <<  "torque_1" << "\n";   

    /////////////////////////////////////////////////////
    /////  avoid large motion at starting points  ///////
    /////////////////////////////////////////////////////
    run_on = 1;  

    // Catch a Ctrl-C event: 
	void  (*sig_h)(int) = sigint_1_step;    

    // Catch a Ctrl-C event:  
    signal(SIGINT, sig_h);    
 
    // dist > dist_threshold && initial_index < max_index
    while(run_on)  
    {
        theta_t = motor_3.read_single_turn(1) - theta_initial;    
        printf("theta_t: %f\n", theta_t);   

        ////////////////////////////////////////////////////////
        // Save Data
        ////////////////////////////////////////////////////////

        dist = sqrt(pow((theta_t - theta_target), 2));    

        /////////////////////////////////////////////////////
        // calculate torque control command 
        ///////////////////////////////////////////////////// 
        torque = clip(-1 * stiffness * (theta_target - theta_t) - damping * (d_theta_e - d_theta_t), torque_lower_bound, torque_upper_bound) * ctl_ratio;   
        curr_t = -1 * stiffness * (theta_target - theta_t) - damping * (d_theta_e - d_theta_t); 

        pos_1 = motor_3.set_torque(1, torque, &d_theta_t, &torque_t);   

        OutFileAngle << theta_t << "," << curr_t << "\n";  

    }

    printf("Rotate to target point !!!! \n");   
    OutFileAngle.close(); 
    motor_3.pack_stop_cmd(1);   

    return 1;  
}
