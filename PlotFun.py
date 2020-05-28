import cv2
import matplotlib.pyplot as plt


def im_read_and_show(img_file):
    img = cv2.imread(img_file)  # 读取RGB图片
    plt.imshow(img)
    plt.axis('off')
    plt.show()
