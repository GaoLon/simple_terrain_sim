#include <iostream>
#include <math.h>
#include <random>
#include <eigen3/Eigen/Dense>
#include <ros/ros.h>
#include <ros/package.h>
#include <nav_msgs/Odometry.h>
#include <std_msgs/Empty.h>
#include <visualization_msgs/Marker.h>

#include <pcl/io/ply_io.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/kdtree/kdtree_flann.h>

#include "kimatic_simulator/DiffCMD.h"

using namespace std;

ros::Subscriber tank_cmd_sub;
ros::Publisher  odom_pub, cloud_pub, marker_pub;
ros::Timer cloud_timer;

kimatic_simulator::DiffCMD tank_cmd;
sensor_msgs::PointCloud2 world_cloud_msg;
nav_msgs::Odometry new_odom;
visualization_msgs::Marker diff_marker;
ros::Time begin_time;

pcl::PointCloud<pcl::PointXYZ>::Ptr world_cloud(new pcl::PointCloud<pcl::PointXYZ>());
pcl::PointCloud<pcl::PointXY>::Ptr world_cloud_plane(new pcl::PointCloud<pcl::PointXY>());
pcl::KdTreeFLANN<pcl::PointXYZ> kd_tree;
pcl::KdTreeFLANN<pcl::PointXY> kd_tree_plane;
std::string map_name;

Eigen::Vector3d now_p;
Eigen::Matrix3d now_R;
double p_init_x, p_init_y, lidar_z, yaw_init, p_init_z;
double time_resolution = 0.01;
bool rcv_cmd = false;

Eigen::Vector4d getTPM(Eigen::Vector3d pos, vector<Eigen::Vector3d> points)
{
	Eigen::Vector4d tpm;

	Eigen::Vector3d mean_points = Eigen::Vector3d::Zero();
	for (size_t i=0; i<points.size(); i++)
		mean_points+=points[i];

	mean_points /= (double)points.size();

	Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
	for (size_t i=0; i<points.size(); i++)
	{
		Eigen::Vector3d v = points[i] - mean_points;
		cov += v * v.transpose();
	}
	cov /= (double)points.size();
	Eigen::EigenSolver<Eigen::Matrix3d> es(cov);
	Eigen::Matrix<double, 3, 1> D = es.pseudoEigenvalueMatrix().diagonal();// eigenValue
	Eigen::Matrix3d V = es.pseudoEigenvectors();    // eigenVector
	Eigen::MatrixXd::Index evalsMax;
	D.minCoeff(&evalsMax);
	Eigen::Matrix<double, 3, 1> n = V.col(evalsMax);
	n.normalize();
	if (n(2, 0) < 0.0)
		n = -n;

	tpm[3] = D(evalsMax) / D.sum() * 3.0;
	if (isnan(tpm[3]))
		n = Eigen::Vector3d(1.0, 0.0, 0.0);
	tpm[3] = mean_points.z();
	tpm[0] = n(0, 0);
	tpm[1] = n(1, 0);
	tpm[2] = n(2, 0);

	return tpm;
}

void getSE3(const Eigen::Vector3d& se2_odom, 
			Eigen::Vector3d& p,
			Eigen::Matrix3d& R)
{
	p(0) = se2_odom(0);
	p(1) = se2_odom(1);
	vector<int> Idxs;
	vector<float> SquaredDists;
	pcl::PointXY pxy;
	pxy.x = se2_odom(0);
	pxy.y = se2_odom(1);
	if (kd_tree_plane.nearestKSearch(pxy, 1, Idxs, SquaredDists) > 0)
	{
		double may_z = world_cloud->points[Idxs[0]].z;
		if (fabs(may_z - p.z()) < 0.3)
			p(2) = may_z;
		else
			;
			// ROS_WARN("may not be the same layer");
	}
	else
		ROS_ERROR("no points in the map");
	vector<Eigen::Vector3d> points;
	pcl::PointXYZ pt;
	pt.x = se2_odom(0);
	pt.y = se2_odom(1);
	pt.z = p(2);
	if (kd_tree.radiusSearch(pt, 0.5, Idxs, SquaredDists) > 0)
	{
		for (size_t i=0; i<Idxs.size(); i++)
			points.emplace_back(Eigen::Vector3d(world_cloud->points[Idxs[i]].x, \
												world_cloud->points[Idxs[i]].y, \
												world_cloud->points[Idxs[i]].z ));
	}
	if (points.empty())
	{	
		Eigen::Quaterniond q(cos(se2_odom(2)), 0.0, 0.0, sin(se2_odom(2)));
		R = q.toRotationMatrix();
	}
	else
	{
		Eigen::Vector4d tpm = getTPM(p, points);
		Eigen::Vector3d xyaw(cos(se2_odom(2)), sin(se2_odom(2)), 0.0);
		Eigen::Vector3d zb = tpm.head(3);
		Eigen::Vector3d yb = zb.cross(xyaw).normalized();
		Eigen::Vector3d xb = yb.cross(zb);
		R.col(0) = xb;
		R.col(1) = yb;
		R.col(2) = zb;
		p(2) = tpm(3);
	}
	return;
}

void normYaw(double& th)
{
	while (th > M_PI)
		th -= M_PI * 2;
	while (th < -M_PI)
		th += M_PI * 2;
}

double calYawFromR(Eigen::Matrix3d R)
{
	Eigen::Vector2d p(R(0, 2), R(1, 2));
	Eigen::Vector2d b(R(0, 0), R(1, 0));
	Eigen::Vector2d x = (Eigen::Matrix2d::Identity()+p*p.transpose()/(1.0-p.squaredNorm()))*b;
	return atan2(x(1), x(0));
}

void initParams()
{
	// global map
	pcl::console::setVerbosityLevel(pcl::console::L_ALWAYS);

	pcl::PCDReader pcd_reader;

	if (pcl::io::loadPCDFile<pcl::PointXYZ>(ros::package::getPath("kimatic_simulator")+"/maps/"+map_name, *world_cloud)==-1) // load global map
	{
		printf("\nCouldn't read %sfile.\n\n",map_name);
	}

	world_cloud->width = world_cloud->points.size();
	world_cloud->height = 1;
	world_cloud->is_dense = true;
	world_cloud->header.frame_id = "map";
	kd_tree.setInputCloud(world_cloud);
	pcl::toROSMsg(*world_cloud, world_cloud_msg);

	for (size_t i=0; i<world_cloud->points.size(); i++)
	{
		pcl::PointXY p;
		p.x = world_cloud->points[i].x;
		p.y = world_cloud->points[i].y;
		world_cloud_plane->points.push_back(p);
	}
	world_cloud_plane->width = world_cloud_plane->points.size();
	world_cloud_plane->height = 1;
	world_cloud_plane->is_dense = true;
	world_cloud_plane->header.frame_id = "map";
	kd_tree_plane.setInputCloud(world_cloud_plane);

	// find initial position
	now_p.z() = p_init_z;
	getSE3(Eigen::Vector3d(p_init_x, p_init_y, yaw_init), now_p, now_R);
	getSE3(Eigen::Vector3d(now_p.x(), now_p.y(), calYawFromR(now_R)), now_p, now_R);
	new_odom.header.stamp    = ros::Time::now();
	new_odom.header.frame_id = "map";
	Eigen::Vector3d lidar_odom = now_p + now_R.col(2) * lidar_z;
	new_odom.pose.pose.position.x = lidar_odom.x();
	new_odom.pose.pose.position.y = lidar_odom.y();
	new_odom.pose.pose.position.z = lidar_odom.z();
	Eigen::Quaterniond q(now_R);
	new_odom.pose.pose.orientation.w = q.w();
	new_odom.pose.pose.orientation.x = q.x();
	new_odom.pose.pose.orientation.y = q.y();
	new_odom.pose.pose.orientation.z = q.z();
	new_odom.twist.twist.linear.x = 0.0;
	new_odom.twist.twist.linear.y = 0.0;
	new_odom.twist.twist.linear.z = 0.0;
	new_odom.twist.twist.angular.x = 0.0;
	new_odom.twist.twist.angular.y = 0.0;
	new_odom.twist.twist.angular.z = 0.0;

	//diff_marker
	diff_marker.header.frame_id = "map";
	diff_marker.id = 0;
	diff_marker.type = visualization_msgs::Marker::MESH_RESOURCE;
	diff_marker.action = visualization_msgs::Marker::ADD;
	diff_marker.pose = new_odom.pose.pose;
	diff_marker.pose.position.x = now_p(0);
	diff_marker.pose.position.y = now_p(1);
	diff_marker.pose.position.z = now_p(2);
	diff_marker.color.a = 1.0;
	diff_marker.color.r = 0.5;
	diff_marker.color.g = 0.5;
	diff_marker.color.b = 0.5;
	diff_marker.scale.x = 0.35;
	diff_marker.scale.y = 0.35;
	diff_marker.scale.z = 0.35;
	diff_marker.mesh_resource = "package://kimatic_simulator/meshes/diff.dae";
}

void rcvVelCmdCallBack(const kimatic_simulator::DiffCMD cmd)
{	
	rcv_cmd 	= true;
	tank_cmd    = cmd;
}

void odomPubCallBack(const ros::TimerEvent& event)
{

	if ((ros::Time::now() - begin_time).toSec() > 5.0)
		cloud_timer.stop();
	if (!rcv_cmd)
	{
		new_odom.header.stamp = ros::Time::now();
    	odom_pub.publish(new_odom);
		diff_marker.header.stamp = ros::Time::now();
		marker_pub.publish(diff_marker);
		return;
	}

	double vx = tank_cmd.velocity;
	double vy = 0.0;
	double wz = tank_cmd.angle_velocity;
	
	Eigen::Vector3d xb = now_R.col(0);
	Eigen::Vector3d yb = now_R.col(1);
	Eigen::Vector3d zb = now_R.col(2);
	Eigen::Vector3d omega = zb * wz;
	Eigen::Vector3d vel = now_R*Eigen::Vector3d(vx, vy, 0.0);

	double cdyaw = cos(wz*time_resolution);
	double sdyaw = sin(wz*time_resolution);
	Eigen::Matrix3d dR;
	dR << cdyaw, -sdyaw, 0.0, \
			sdyaw, cdyaw , 0.0, \
			0.0  , 0.0   , 1.0;
	now_p = now_p + (xb*vx + yb*vy) * time_resolution;
	now_R = now_R*dR;
	
	// ros::Time t = ros::Time::now();
	getSE3(Eigen::Vector3d(now_p.x(), now_p.y(), calYawFromR(now_R)), now_p, now_R);
	Eigen::Quaterniond q(now_R);
	new_odom.header.stamp = ros::Time::now();
	Eigen::Vector3d lidar_odom = now_p + now_R.col(2) * lidar_z;
	new_odom.pose.pose.position.x  = lidar_odom.x();
	new_odom.pose.pose.position.y  = lidar_odom.y();
	new_odom.pose.pose.position.z  = lidar_odom.z();
	new_odom.pose.pose.orientation.w  = q.w();
	new_odom.pose.pose.orientation.x  = q.x();
	new_odom.pose.pose.orientation.y  = q.y();
	new_odom.pose.pose.orientation.z  = q.z();
	new_odom.twist.twist.linear.x  = vel(0);
	new_odom.twist.twist.linear.y  = vel(1);
	new_odom.twist.twist.linear.z  = vel(2);
	new_odom.twist.twist.angular.x = omega(0);
	new_odom.twist.twist.angular.y = omega(1);
	new_odom.twist.twist.angular.z = omega(2);
	diff_marker.header.stamp = ros::Time::now();
	diff_marker.pose.orientation = new_odom.pose.pose.orientation;
	diff_marker.pose.position.x = now_p.x();
	diff_marker.pose.position.y = now_p.y();
	diff_marker.pose.position.z = now_p.z();
	odom_pub.publish(new_odom);
	marker_pub.publish(diff_marker);
}

void cloudCallBack(const ros::TimerEvent& event)
{
	cloud_pub.publish(world_cloud_msg);
}

int main (int argc, char** argv) 
{        
    ros::init (argc, argv, "ugv_kinematic_model_node");
    ros::NodeHandle nh( "~" );

    nh.param("init_x", p_init_x, 1.0);
    nh.param("init_y", p_init_y, 1.0);
    nh.param("p_init_z", p_init_z, 1.0);
    nh.param("lidar_z", lidar_z, 1.0);
	nh.param("init_yaw", yaw_init, 0.0);
	nh.param("map_name", map_name, std::string("quick.pcd"));

	initParams();

	tank_cmd_sub = nh.subscribe( "command", 1, rcvVelCmdCallBack );
    odom_pub  = nh.advertise<nav_msgs::Odometry>("odometry", 1);
	marker_pub = nh.advertise<visualization_msgs::Marker>("marker", 1);
    cloud_pub  = nh.advertise<sensor_msgs::PointCloud2>("/global_cloud", 1);
	cloud_timer = nh.createTimer(ros::Duration(2.0), cloudCallBack);
	ros::Timer odom_timer = nh.createTimer(ros::Duration(time_resolution), odomPubCallBack);
	begin_time = ros::Time::now();

	ros::spin();
    return 0;
}