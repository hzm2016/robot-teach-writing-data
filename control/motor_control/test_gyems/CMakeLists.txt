###################################################################
# Data file macros generator:
###################################################################


###################################################################
# Control program - basic functions:
###################################################################

add_executable(test_gyems
    test.cpp)  

target_link_libraries(test_gyems LINK_PUBLIC
m
rtsetup_lib
rtutils_lib
thread_service_lib
daq_lib
socketcan_lib
gyems_can_lib
renishaw_can_lib
s526_lib
${CMAKE_THREAD_LIBS_INIT}) 
