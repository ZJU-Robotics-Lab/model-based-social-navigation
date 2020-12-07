#!/usr/bin/env python
import rospy
from std_msgs.msg import String
from gazebo_msgs.msg import ModelStates

"""
string[] name                                                                                                                   
geometry_msgs/Pose[] pose                                                                                                       
  geometry_msgs/Point position                                                                                                  
    float64 x                                                                                                                   
    float64 y                                                                                                                   
    float64 z                                                                                                                   
  geometry_msgs/Quaternion orientation                                                                                          
    float64 x                                                                                                                   
    float64 y                                                                                                                   
    float64 z                                                                                                                   
    float64 w                                                                                                                   
geometry_msgs/Twist[] twist                                                                                                     
  geometry_msgs/Vector3 linear                                                                                                  
    float64 x                                                                                                                   
    float64 y                                                                                                                   
    float64 z                                                                                                                   
  geometry_msgs/Vector3 angular                                                                                                 
    float64 x                                                                                                                   
    float64 y                                                                                                                   
    float64 z                                                                                                                   
                  
"""

class GazeboModel(object):
    def __init__(self, robots_name_list = ['mobile_base_2', 'mobile_base_1']):
    
        # We wait for the topic to be available and when it is then retrive the index of each model
        # This was separated from callbal to avoid doing this in each callback
        self._robots_models_dict = {}
        self._robots_pose_list = []
        self._robots_index_dict = {}
        self._robots_name_list = robots_name_list
        
        self.get_robot_index()
        
        # We now start the suscriber once we have the indexes of each model
        rospy.Subscriber("/gazebo/model_states", ModelStates, self.callback)
    
    def get_robot_index(self):
        
        data = None
        while data is None:
            rospy.loginfo("Retrieveing Model indexes ")
            try:
                data = rospy.wait_for_message("/gazebo/model_states", ModelStates, timeout=5)
                # Save it in the format {"robot1":4,"robot2":2}
                for robot_name in self._robots_name_list:
                    index = data.name.index(robot_name)
                    rospy.loginfo("indexes ="+str(index))
                    self._robots_index_dict[robot_name] = index
            
            except Exception as e:
                s = str(e)
                rospy.loginfo("Error = "+ s)
                #pass
            
        #rospy.loginfo("Final robots_index_dict =  %s ", str(self._robots_index_dict))
            
    def callback(self,data):
        #rospy.loginfo(rospy.get_caller_id() + "I heard %s", data.name)
        for robot_name in self._robots_name_list:
            # Retrieve the corresponding index
            data_index = self._robots_index_dict[robot_name]
            # Get the pose data from theat index
            data_pose = data.pose[data_index]
            # Save the pose inside the dict {"robot1":pose1,"robot2":pose2}
            self._robots_models_dict[robot_name] = data_pose
            
            #rospy.loginfo("%s pose orientation Z =  %s ", robot_name ,str(data_pose.orientation.z))
            #rospy.loginfo("Final _robots_models_dict =  %s ", str(self._robots_models_dict))
            
    def get_model_pose(self,robot_name):
        
        pose_now = None
        
        try:
            pose_now = self._robots_models_dict[robot_name]
            #rospy.loginfo("robots_models_dict =  %s ", str(self._robots_models_dict))
        except Exception as e:
            s = str(e)
            rospy.loginfo("Error, The _robots_models_dict is not ready = "+ s)
        
        #rospy.loginfo("Final _robots_models_dict =  %s ", str(self._robots_models_dict))
        #rospy.loginfo("Final pose_now =  %s ", str(self.pose_now))
        return pose_now


def listener():
    rospy.init_node('listener', anonymous=True)
    gz_model = GazeboModel()
    rate = rospy.Rate(1) # 10hz
    while not rospy.is_shutdown():
        gz_model.get_model_pose(robot_name="mobile_base_1")
        rate.sleep()
    #rospy.spin()

if __name__ == '__main__':
    listener()