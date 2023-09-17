import math
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def split_Prokudin_Gorskii_channels(img: np.ndarray) -> list[np.ndarray]:
    # edge_crop_size = 10
    (img_y, _) = img.shape
    img_y_size = math.floor(img_y/3)
    img_channels = [
        img[0:img_y_size, :],
        img[img_y_size:2*img_y_size, :],
        img[2*img_y_size:3*img_y_size, :]
    ]
    return img_channels
    return [
        cv.copyMakeBorder(channel,10,10,10,10, cv.BORDER_CONSTANT, value=255) for channel in img_channels
    ]

def get_edges(img: np.ndarray) -> np.ndarray:
    edges = cv.Laplacian(img, -1)
    return edges

def allign_edges(R: np.ndarray, G: np.ndarray, B: np.ndarray):
    stride_area = 5
    base_channel = cv.copyMakeBorder(R, stride_area,stride_area,stride_area,stride_area, cv.BORDER_CONSTANT, value=0)
    # objetive_channel = G
    best_similarity = 0
    (best_x, best_y) = (0, 0)
    for i in range(stride_area+1):
        for j in range(stride_area+1):
            objetive_channel = cv.copyMakeBorder(G, j, stride_area-j, i, stride_area-i, cv.BORDER_CONSTANT, value=0)
            a = base_channel.flatten()
            b = objetive_channel.flatten()
            print(base_channel.shape, objetive_channel.shape)
            print(a.shape, b.shape)
            # similarity = np.dot(a, b)
            # if similarity > best_similarity:
            #     (best_x, best_y) = (i, j)
            # plt.subplot(121)
            # plt.imshow(base_channel)
            # plt.axis('off')
            # plt.subplot(122)
            # plt.imshow(objetive_channel)
            # plt.axis('off')
            # plt.show()
    print(best_x, best_y)
    pass

def compose_channels(R: np.ndarray, G: np.ndarray, B: np.ndarray) -> np.ndarray:
    img = cv.merge([R, G, B])
    return img

def main():
    img = cv.imread("HW1/img/img_M.jpg", cv.IMREAD_GRAYSCALE)
    channels = split_Prokudin_Gorskii_channels(img)
    edges = [get_edges(channel) for channel in channels]
    alligned_edges = allign_edges(edges[0], edges[1], edges[2])
    color_img = compose_channels(channels[2], channels[1], channels[0])

    grid_size = (3,4)
    
    ax1 = plt.subplot2grid(grid_size, (0,0), rowspan=3)
    ax1.imshow(img, cmap='gray')
    plt.axis('off')
    
    ax2 = plt.subplot2grid(grid_size, (0,1))
    ax2.imshow(channels[0], cmap = 'gray')
    plt.axis('off')
    ax3 = plt.subplot2grid(grid_size, (1,1))
    ax3.imshow(channels[1], cmap = 'gray')
    plt.axis('off')
    ax4 = plt.subplot2grid(grid_size, (2,1))
    ax4.imshow(channels[2], cmap = 'gray')
    plt.axis('off')
    
    ax5 = plt.subplot2grid(grid_size, (0,2))
    ax5.imshow(edges[0], cmap = 'gray')
    plt.axis('off')
    ax6 = plt.subplot2grid(grid_size, (1,2))
    ax6.imshow(edges[1], cmap = 'gray')
    plt.axis('off')
    ax7 = plt.subplot2grid(grid_size, (2,2))
    ax7.imshow(edges[2], cmap = 'gray')
    plt.axis('off')

    ax8 = plt.subplot2grid(grid_size, (1,3))
    ax8.imshow(color_img)
    plt.axis('off')
    
    plt.show()

if __name__ == '__main__':
    main()