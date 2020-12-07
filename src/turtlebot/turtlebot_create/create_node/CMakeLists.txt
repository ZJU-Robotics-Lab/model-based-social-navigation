cmake_minimum_required(VERSION 2.8.3)
project(create_node)

find_package(catkin REQUIRED COMPONENTS create_driver
                                        diagnostic_msgs
                                        dynamic_reconfigure
                                        geometry_msgs
                                        message_generation
                                        nav_msgs
                                        rospy
                                        tf)

catkin_python_setup()

add_message_files(DIRECTORY msg
                  FILES BatteryState.msg
                        Drive.msg
                        RawTurtlebotSensorState.msg
                        RoombaSensorState.msg
                        Turtle.msg
                        TurtlebotSensorState.msg)
                        
add_service_files(DIRECTORY srv
                  FILES SetDigitalOutputs.srv
                        SetTurtlebotMode.srv)

generate_messages(DEPENDENCIES diagnostic_msgs
                               geometry_msgs
                               nav_msgs)
                               
generate_dynamic_reconfigure_options(cfg/TurtleBot.cfg)

catkin_package(CATKIN_DEPENDS diagnostic_msgs
                              geometry_msgs
                              message_runtime
                              nav_msgs)

install(PROGRAMS scripts/kinect_breaker_enabler.py
                 scripts/load_calib.py
                 nodes/turtlebot_node.py
        DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
