# Create a library
add_library (s526_lib model526.cpp) 
# target_link_libraries(s526_lib LINK_PUBLIC
# gyems_can_lib renishaw_can_lib) 

target_include_directories(s526_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
