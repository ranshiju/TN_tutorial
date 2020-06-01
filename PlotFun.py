import cv2
import matplotlib.pyplot as plt
import torch as tc


def im_read_and_show(img_file):
    img = cv2.imread(img_file)  # 读取RGB图片
    plt.imshow(img)
    plt.axis('off')
    plt.show()


def plot(x, *y, marker='s'):
    if type(x) is tc.Tensor:
        if x.device != 'cpu':
            x = x.cpu()
        x = x.numpy()
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    if len(y) > 0.5:
        for y0 in y:
            if type(y0) is tc.Tensor:
                if y0.device != 'cpu':
                    y0 = y0.cpu()
                y0 = y0.numpy()
            ax.plot(x, y0, marker=marker)
    else:
        ax.plot(x, marker=marker)
    plt.show()
