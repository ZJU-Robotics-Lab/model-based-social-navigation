// map, odom, tf

#include <tf/transform_broadcaster.h>
#include <nav_msgs/Odometry.h>

#include <gazebo_msgs/ModelState.h>
#include <gazebo_msgs/SetModelState.h>
#include <gazebo_msgs/GetModelState.h>
#include <nav_msgs/Odometry.h>

#include <geometry_msgs/PoseStamped.h>

ros::Time current_time;

// void odom_callback(const nav_msgs::Odometry& Odometry_msg){

//     current_time = Odometry_msg.header.stamp;

// }

int main(int argc, char **argv)
{

    ros::init(argc, argv, "map_odom_publisher");
    ros::NodeHandle n;

    tf::TransformBroadcaster map_broadcaster; // 加载地图时使用

    // ros::Subscriber odom_sub = n.subscribe("odom",1,odom_callback);

    
    current_time = ros::Time::now();

    ros::Rate r(20.0);
    while (n.ok())
    {

        ros::spinOnce(); // check for incoming messages


        // 可以改为在launch文件中实现
        geometry_msgs::Quaternion map_quat = tf::createQuaternionMsgFromYaw(0);
        current_time = ros::Time::now();
        geometry_msgs::TransformStamped map_trans;
        map_trans.header.stamp = current_time;
        map_trans.header.frame_id = "map";
        map_trans.child_frame_id = "odom";
        map_trans.transform.translation.x = 0.0;
        map_trans.transform.translation.y = 0.0;
        map_trans.transform.translation.z = 0.0;
        map_trans.transform.rotation = map_quat;
        map_broadcaster.sendTransform(map_trans);


        r.sleep();
    }
}
