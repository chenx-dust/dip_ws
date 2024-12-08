import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped

last_odom = None

def odom_callback(msg):
    global last_odom
    last_odom = msg

def odom_combined_callback(msg: PoseWithCovarianceStamped):
    # 修改接收到的 Odometry 消息中的 frame_id
    # msg.header.frame_id = "odom_combined"  # 修改为目标坐标系
    if last_odom is None:
        return
    odom_msg = Odometry()
    odom_msg.header = msg.header
    odom_msg.header.frame_id = "odom_combined"
    odom_msg.pose = msg.pose
    odom_msg.twist = last_odom.twist
    
    # 这里可以进行其他处理，比如发布到新的话题
    odom_pub.publish(odom_msg)

def listener():
    rospy.init_node('odom_frame_trans_node', anonymous=True)
    
    # 创建 Odometry 消息的订阅者
    rospy.Subscriber("/odom", Odometry, odom_callback)
    rospy.Subscriber("/odom_combined", PoseWithCovarianceStamped, odom_combined_callback)
    
    # 创建一个新的发布者，将修改后的消息发布到另一个话题
    global odom_pub
    odom_pub = rospy.Publisher("/odom_modified", Odometry, queue_size=10)

    rospy.spin()

if __name__ == '__main__':
    listener()
