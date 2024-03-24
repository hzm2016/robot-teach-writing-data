# Create a library
add_library (sea_model_lib sea_model.cpp) 
target_link_libraries(sea_model_lib LINK_PUBLIC
gyems_can_lib renishaw_can_lib) 

target_include_directories(sea_model_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
