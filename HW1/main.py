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

def get_edges(img: np.ndarray) -> np.ndarray:
    # edges = cv.Laplacian(img, -1)
    edges = cv.Sobel(img, ddepth=-1, dx=1, dy=0)
    def binary_edge(x): 
        edge_threshold = 155
        if x < edge_threshold:
            return 0
        else:
            return 255
    edges = np.vectorize(binary_edge)(edges)
    return edges

def allign_edges(R: np.ndarray, G: np.ndarray, B: np.ndarray):
    stride_area = 20
    base_channel = cv.copyMakeBorder(R, stride_area,stride_area,stride_area,stride_area, cv.BORDER_CONSTANT, value=0)
    best_similarity_G = 0
    best_similarity_B = 0
    (best_x_G, best_y_G) = (0, 0)
    (best_x_B, best_y_B) = (0, 0)
    for i in range((2*stride_area)+1):
        for j in range((2*stride_area)+1):
            objetive_channel_G = cv.copyMakeBorder(G, j, (2*stride_area)-j, i, (2*stride_area)-i, cv.BORDER_CONSTANT, value=0)
            objetive_channel_B = cv.copyMakeBorder(B, j, (2*stride_area)-j, i, (2*stride_area)-i, cv.BORDER_CONSTANT, value=0)
            flat_R = base_channel.flatten()
            flat_G = objetive_channel_G.flatten()
            flat_B = objetive_channel_B.flatten()
            similarity_G = np.dot(flat_R, flat_G)
            similarity_B = np.dot(flat_R, flat_B)
            if similarity_G > best_similarity_G:
                best_similarity_G = similarity_G
                (best_x_G, best_y_G) = (i, j)
            if similarity_B > best_similarity_B:
                best_similarity_B = similarity_B
                (best_x_B, best_y_B) = (i, j)
    return {
        'R_padding': (stride_area, stride_area, stride_area, stride_area),
        'G_padding': (best_y_G, (2*stride_area)-best_y_G, best_x_G, (2*stride_area)-best_x_G),
        'B_padding': (best_y_B, (2*stride_area)-best_y_B, best_x_B, (2*stride_area)-best_x_B)
    }

def compose_channels(R: np.ndarray, G: np.ndarray, B: np.ndarray) -> np.ndarray:
    img = cv.merge([R, G, B])
    return img

def main():
    img = cv.imread("HW1/img/img_M.jpg", cv.IMREAD_GRAYSCALE)
    channels = split_Prokudin_Gorskii_channels(img)
    edges = [get_edges(channel) for channel in channels]
    channel_paddings = allign_edges(edges[0], edges[1], edges[2])
    alligned_edges = [
        cv.copyMakeBorder(edges[0], 
                          channel_paddings['B_padding'][0], 
                          channel_paddings['B_padding'][1],
                          channel_paddings['B_padding'][2],
                          channel_paddings['B_padding'][3],
                          cv.BORDER_CONSTANT, value=0),
        cv.copyMakeBorder(edges[1], 
                          channel_paddings['G_padding'][0], 
                          channel_paddings['G_padding'][1],
                          channel_paddings['G_padding'][2],
                          channel_paddings['G_padding'][3],
                          cv.BORDER_CONSTANT, value=0),
        cv.copyMakeBorder(edges[2], 
                          channel_paddings['R_padding'][0], 
                          channel_paddings['R_padding'][1],
                          channel_paddings['R_padding'][2],
                          channel_paddings['R_padding'][3],
                          cv.BORDER_CONSTANT, value=0),
    ]
    alligned_channels = [
        cv.copyMakeBorder(channels[0], 
                          channel_paddings['B_padding'][0], 
                          channel_paddings['B_padding'][1],
                          channel_paddings['B_padding'][2],
                          channel_paddings['B_padding'][3],
                          cv.BORDER_CONSTANT, value=0),
        cv.copyMakeBorder(channels[1], 
                          channel_paddings['G_padding'][0], 
                          channel_paddings['G_padding'][1],
                          channel_paddings['G_padding'][2],
                          channel_paddings['G_padding'][3],
                          cv.BORDER_CONSTANT, value=0),
        cv.copyMakeBorder(channels[2], 
                          channel_paddings['R_padding'][0], 
                          channel_paddings['R_padding'][1],
                          channel_paddings['R_padding'][2],
                          channel_paddings['R_padding'][3],
                          cv.BORDER_CONSTANT, value=0),
    ]
    color_edges = compose_channels(alligned_edges[2], alligned_edges[1], alligned_edges[0])
    color_img = compose_channels(alligned_channels[2], alligned_channels[1], alligned_channels[0])

    grid_size = (3,5)
    
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
    ax8.imshow(color_edges)
    plt.axis('off')

    ax9 = plt.subplot2grid(grid_size, (1,4))
    ax9.imshow(color_img)
    plt.axis('off')
    
    plt.show()

if __name__ == '__main__':
    main()