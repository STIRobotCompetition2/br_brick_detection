import cv2
import torch
import numpy as np
from rclpy.node import Node
import rclpy.exceptions
import rclpy
from sensor_msgs.msg import Image, CompressedImage
from visualization_msgs.msg import Marker, MarkerArray
# from std_srvs.srv import Trigger
# from br_brick_management.msg import PolygonArrayStamped
from cv_bridge import CvBridge
import copy
import rclpy.parameter
import os
import rclpy.qos as QoS

class YoloDetector(Node):
    def __init__(self, name:str = "brick_detection"):
        super().__init__(name)
        
        self.declare_parameter("input_image_topic", "")
        self.declare_parameter("output_image_topic", "")
        self.declare_parameter("model_dir", "")
        self.declare_parameter("model_name", "")
        self.declare_parameter("yolov5_dir", "")


        input_image_topic = self.get_parameter_or("input_image_topic", "/camera/image").get_parameter_value().string_value
        output_image_topic = self.get_parameter_or("output_image_topic", "/brick_detection/bounding_boxes").get_parameter_value().string_value
        
        model_dir = self.get_parameter("model_dir").get_parameter_value().string_value
        model_name = self.get_parameter("model_name").get_parameter_value().string_value
        yolov5_dir = self.get_parameter("yolov5_dir").get_parameter_value().string_value

        self.pub_annotated_images = self.get_parameter_or("pub_annotated_images", True)

        self.cv_bridge = CvBridge()
     
        try:
            self.get_logger().info(yolov5_dir)
            self.get_logger().info(model_name)
            self.get_logger().info(model_dir)

            self.model = torch.hub.load(yolov5_dir, 'custom', path=os.path.join(model_dir, model_name), source='local')
        except Exception as e:
            raise rclpy.exceptions.ParameterException("Cannot create model from {} with model-parameter file {}/{}.\n:{}".format(yolov5_dir, model_dir, model_name, e.with_traceback()))

        if self.pub_annotated_images:

            self.annotated_image_pub = self.create_publisher(CompressedImage, output_image_topic, qos_profile=QoS.qos_profile_sensor_data)
        
        self.input_image_sub = self.create_subscription(Image, input_image_topic, self.image_callback, qos_profile=QoS.qos_profile_sensor_data)
        self.output_marker_pub = self.create_publisher(MarkerArray, "/detected_markers", qos_profile=15)

        
        self.get_logger().info("Yolo Brick Detection initialized successfully. Awaiting images on topic {}".format(input_image_topic))


    def image_callback(self, msg:Image):
        valid_image = self.preprocess_image(msg)
        
        image_annotated = self.inference(valid_image)
        
        self.publish(None, image_annotated)
        self.get_logger().debug("Yolo Brick Detection processed image")

    def preprocess_image(self, input:Image): 
        # self.get_logger().info("Unpack")

        input_as_numpy = self.cv_bridge.imgmsg_to_cv2(input, desired_encoding='passthrough')
        # if not (input_as_numpy.shape[0] == 320 and input_as_numpy.shape[1] == 320):
        #     self.get_logger().warn("Yolo Brick Detection got image of resolution {} but requires 320x320. Cropping it ...".format(input_as_numpy.shape[0:2]))
        #     square_side_length = np.min(input_as_numpy.shape[0:2])
        #     return cv2.resize(input_as_numpy[:square_side_length,:square_side_length,:], dsize=(320, 320,), interpolation=cv2.INTER_CUBIC)
        return input_as_numpy

    def inference(self, input:np.ndarray):
        # self.get_logger().info("Cut")

        output = np.zeros(input.shape[0:2], dtype=np.uint8)
        print(input.shape)
        # self.get_logger().info("Run")

        for x_idx in range(3):
            for y_idx in range(2):
                temp = input[160 * y_idx:320+160 * (y_idx), 160 * x_idx:320+160 * (x_idx)]
                # self.get_logger().info("Inf start")

                yolo_output = self.model(temp, size=320, )
                # self.get_logger().info("Inf stop")
                
                for detection in yolo_output.xyxy[0]:
                    if detection[4] < 0.4: continue
                    corners = detection[0:4].tolist()
                    corners = [int(val) for val in corners]
                    cv2.rectangle(temp, corners[0:2], corners[2:4], [255,255,0],thickness=5)
                    input[160 * y_idx:320+160 * (y_idx), 160 * x_idx:320+160 * (x_idx)] = temp
                    corners[1] += 160 * y_idx
                    corners[3] += 160 * y_idx
                    corners[0] += 160 * x_idx
                    corners[2] += 160 * x_idx

                    cv2.rectangle(output, corners[0:2], corners[2:4], 255,-1)
                # self.get_logger().info("Draw Stop")

        
        # for x_idx in range(0):
        #     for y_idx in range(0):
        #         temp = input[160 + 320 * y_idx:160 + 320 * (y_idx + 1), 160+320 * x_idx:160+320 * (x_idx + 1)]
        #         yolo_output = self.model(temp, size=320)
        #         for detection in yolo_output.xyxy[0]:
        #             if detection[4] < 0.4: continue
        #             corners = detection[0:4].tolist()
        #             corners = [int(val) for val in corners]
        #             cv2.rectangle(temp, corners[0:2], corners[2:4], [255,255,0],thickness=5)
        #             input[160 + 320 * y_idx:160 + 320 * (y_idx + 1), 160+320 * x_idx:160+320 * (x_idx + 1)] = temp
        #             corners[1] += 160 + 320 * y_idx
        #             corners[3] += 160 + 320 * y_idx
        #             corners[0] += 160+320 * x_idx
        #             corners[2] += 160+320 * x_idx
        #             cv2.rectangle(output, corners[0:2], corners[2:4], 255,-1)
        output = cv2.morphologyEx(output, cv2.MORPH_DILATE, (3,3,))
        contours, hierarchy = cv2.findContours(output,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)[-2:]
        idx = 0 
        output_ma = MarkerArray()
        stamp = self.get_clock().now()
        
        for cnt in contours:
            idx += 1
            x,y,w,h = cv2.boundingRect(cnt)
            input = cv2.rectangle(input, [x,y],[x+w, y+h], [255,0,255], thickness=3)
            center_x = (y + h/2) * -0.00154078 + 0.367264
            center_y = (x + w/2) * -0.00146299 + 0.406898
            ma = Marker()
            ma.header.frame_id = "base_link"
            ma.header.stamp = stamp.to_msg()
            ma.color.r = 1.0
            ma.color.a = 1.0
            ma.scale.x = 0.05
            ma.scale.y = 0.05
            ma.scale.z = 0.05
            ma.type = Marker.SPHERE
            ma.action = Marker.ADD
            ma.pose.position.x = center_x
            ma.pose.position.y = center_y
            ma.pose.position.z = 0.
            ma.pose.orientation.w = 1.0
            ma.id = idx
            output_ma.markers.append(ma)


            self.get_logger().info("X: " + str(center_x) + " Y: " + str(center_y))
            #cv2.rectangle(im,(x,y),(x+w,y+h),(200,0,0),2)
        self.output_marker_pub.publish(output_ma)
        return input
    
    def annotate(self, input:np.ndarray, yolo_output):
        output = copy.deepcopy(input)
        for detection in yolo_output.xyxy[0]:
            if detection[4] < 0.7: continue
            corners = detection[0:4].tolist()
            corners = [int(val) for val in corners]
            cv2.rectangle(output, pt1=corners[0:2], pt2=corners[2:4], color=[255,0,0,255],thickness=3)
        return output

    def publish(self, _, annotated_image = None):
        if annotated_image is not None:
            annotated_image = cv2.resize(annotated_image, (400, 300, ))
            annotated_as_msg = self.cv_bridge.cv2_to_compressed_imgmsg(annotated_image,dst_format="jpeg")
            annotated_as_msg.format = "jpeg"
            self.annotated_image_pub.publish(annotated_as_msg)

def main(args=None):
    rclpy.init(args = args)
    yb = YoloDetector()
    rclpy.spin(yb)

if __name__=="__main__":
    main()

