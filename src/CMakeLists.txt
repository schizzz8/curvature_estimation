add_executable(curvature_estimation_example curvature_estimation_example.cpp)

target_link_libraries(curvature_estimation_example 
  ${PCL_LIBRARIES}
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
)
