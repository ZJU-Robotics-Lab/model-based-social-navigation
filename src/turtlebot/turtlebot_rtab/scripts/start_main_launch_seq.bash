#!/bin/bash         
         echo $1 
         COUNTER=0
         while [  $COUNTER -lt $1 ]; do
             echo The counter is $COUNTER
             let COUNTER=COUNTER+1
             echo $ROS_PACKAGE_PATH;
             ps faux | grep roscore;
             ps faux | grep node;
             sleep 1
         done
         #echo "Source Bashrc"
         #. /home/ubuntu/.bashrc
         echo "Start Main Launch"
         roslaunch turtlebot_rtab main_original.launch gui:=false
