import rospy
from nav_msgs.msg import Odometry

def odom_callback(msg):
    # 修改接收到的 Odometry 消息中的 frame_id
    msg.header.frame_id = "odom_combined"  # 修改为目标坐标系
    
    # 这里可以进行其他处理，比如发布到新的话题
    odom_pub.publish(msg)

def listener():
    rospy.init_node('odom_frame_trans_node', anonymous=True)
    
    # 创建 Odometry 消息的订阅者
    rospy.Subscriber("/odom", Odometry, odom_callback)
    
    # 创建一个新的发布者，将修改后的消息发布到另一个话题
    global odom_pub
    odom_pub = rospy.Publisher("/odom_modified", Odometry, queue_size=10)

    rospy.spin()

if __name__ == '__main__':
    listener()
