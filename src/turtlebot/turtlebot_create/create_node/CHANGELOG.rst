^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for package create_node
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

2.3.1 (2016-06-24)
------------------

2.3.0 (2014-11-30)
------------------
* Set queue size
* Actually use provided configuration
* Contributors: Kenneth Bogert, trainman419

2.2.1 (2014-07-23)
------------------
* Remove open() to fix double-open exception. Improve error info when an exception on opening serial port happens
* Bug fixes
* Attempt to use the usb to serial converter's id
  If the robot receives a velocity command we'll assume that it's been
  brought up and must now move.  If we don't have the kinect's serial number
  yet assume we won't ever get it.  Instead attempt to find a calibration file
  that matches the serial number of the usb to serial converter that should be
  attached at /dev/ttyUSB0.  This setup will prevent the calibration script from
  running indefinitely waiting for a kinect that might never come.
* Import calibration load script
  This script uses the serial number of the attached kinect to identify the turtlebot that is currently connected and attempts to load a calibration file for it from a standard location.
  Needs more work to detect an xtion plugged in instead of a kinect, as well as if neither are plugged in.
* Contributors: Kenneth Bogert, Paul Bouchier

2.2.0 (2013-08-30)
------------------
* adds bugtracker and repo info to package.xml

2.1.0 (2013-07-19)
------------------

* ROS Hydro beta release
* Fully catkinized
