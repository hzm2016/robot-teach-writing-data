# Create a library
add_library (robot_cal_lib robot_cal.cpp)
target_link_libraries(robot_cal_lib LINK_PUBLIC
gyems_can_lib renishaw_can_lib) 

target_include_directories(robot_cal_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}) 
