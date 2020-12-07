#!/usr/bin/env python  
import roslib
roslib.load_manifest('multiple_kobukis')
import rospy
import tf
from kobuki_get_gazebo_pose import GazeboModel
import time

def handle_turtle_pose(pose_msg, robot_name):
    br = tf.TransformBroadcaster()
    
    robot_name_frame = "robot"+robot_name[-1:]+"_tf/odom"
    br.sendTransform((pose_msg.position.x,pose_msg.position.y,pose_msg.position.z),
                     (pose_msg.orientation.x,pose_msg.orientation.y,pose_msg.orientation.z,pose_msg.orientation.w),
                     rospy.Time.now(),
                     robot_name_frame,
                     "/world")

def publisher_of_tf():
    
    rospy.init_node('publisher_of_tf_node', anonymous=True)
    gazebo_model_object = GazeboModel()
    robot_name_list = ["mobile_base_1","mobile_base_2"]
    robot_name="mobile_base_1"
    time.sleep(3)
    for robot_name in robot_name_list:
        pose_now = gazebo_model_object.get_model_pose(robot_name)
    
    rate = rospy.Rate(10) # 10hz
    while not rospy.is_shutdown():
        for robot_name in robot_name_list:
            pose_now = gazebo_model_object.get_model_pose(robot_name)
            if not pose_now:
                print "The Pose is not yet"+str(robot_name)+" available...Please try again later"
            else:
                print "POSE NOW of robot"+str(robot_name)+",==> "+str(pose_now)
                handle_turtle_pose(pose_now, robot_name)
        rate.sleep()
    

if __name__ == '__main__':
    try:
        publisher_of_tf()
    except rospy.ROSInterruptException:
        pass
