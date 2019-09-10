import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import tensorflow as tf
import scipy.misc
import cv2
import math

sess = tf.InteractiveSession()
saver = tf.train.import_meta_graph('./models/linear/model_linear.ckpt.meta')
saver.restore(sess, "./models/linear/model_linear.ckpt")

graph = tf.get_default_graph()
predicted_angle_ln = graph.get_tensor_by_name("predicted_angle_ln:0")
true_image_ln = graph.get_tensor_by_name("true_image_ln:0")
keep_prob_ln = graph.get_tensor_by_name("keep_prob_ln:0")

img = cv2.imread('./images/steering_wheel_image.jpg',0)
rows,cols = img.shape

smoothed_angle = 0


#read data.txt
xs = []
ys = []
with open("data/data.txt") as f:
    for line in f:
        xs.append("data/" + line.split()[0]) 
        ys.append(float(line.split()[1]) * scipy.pi / 180)

#get number of images
num_images = len(xs)


i = math.ceil(num_images*0.7)
print("Starting frame of video:" +str(i))

while(cv2.waitKey(10) != ord('q')):
    full_image = scipy.misc.imread("data/" + str(i) + ".jpg", mode="RGB")
    image = scipy.misc.imresize(full_image[-150:], [66, 200]) / 255.0
    degrees = predicted_angle_ln.eval(feed_dict={true_image_ln: [image], keep_prob_ln: 1.0})[0][0] * 180.0 / scipy.pi
    print("Steering angle: " + str(degrees) + " (pred)\t" + str(ys[i]*180/scipy.pi) + " (actual)")
    cv2.imshow("frame", cv2.cvtColor(full_image, cv2.COLOR_RGB2BGR))
    smoothed_angle += 0.2 * pow(abs((degrees - smoothed_angle)), 2.0 / 3.0) * (degrees - smoothed_angle) / abs(degrees - smoothed_angle)
    M = cv2.getRotationMatrix2D((cols/2,rows/2),-smoothed_angle,1)
    dst = cv2.warpAffine(img,M,(cols,rows))
    cv2.imshow("steering wheel", dst)
    i += 1

cv2.destroyAllWindows()
sess.close()