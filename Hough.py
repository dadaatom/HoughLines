import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import random

def hough(edges, threshold, rhoCount=180, thetaCount=180):
    diagDistance = np.sqrt(np.square(edges.shape[0]) + np.square(edges.shape[1]))

    thetas = np.arange(0, 180, step= 180 / thetaCount)
    rhos = np.arange(-diagDistance, diagDistance, step= (2 * diagDistance) / rhoCount)

    cos_thetas = np.cos(np.deg2rad(thetas))
    sin_thetas = np.sin(np.deg2rad(thetas))

    houghSpace = np.zeros((len(rhos), len(rhos)))
    lines = []

    for y in range(edges.shape[0]):
        for x in range(edges.shape[1]):
            if edges[y][x] != 0:
                edge_point = [y - edges.shape[0]/2, x - edges.shape[1]/2]
                for iTheta in range(len(thetas)):
                    #theta = thetas[iTheta]
                    rho = (edge_point[1] * cos_thetas[iTheta]) + (edge_point[0] * sin_thetas[iTheta])

                    iRho = np.argmin(np.abs(rhos - rho))

                    houghSpace[iRho][iTheta] += 1

    for y in range(houghSpace.shape[0]):
        for x in range(houghSpace.shape[1]):
            if houghSpace[y][x] > threshold:
                rho = rhos[y]
                theta = thetas[x]

                lines.append([[rho, theta], x, y])

    return houghSpace, lines

def displayHoughLines(image, lines):
    if len(lines) > 0:
        for line in lines:
            rho, theta = line[0]

            a = np.cos(math.radians(theta))
            b = np.sin(math.radians(theta))

            x0 = a * rho + image.shape[1]/2
            y0 = b * rho + image.shape[0]/2

            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 1)

    return image

def display(name, image):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, image)


# ===================== IMAGES ===================== #

hough1Path = "Images/hough1.png"
hough2Path = "Images/hough2.png"


# ===================== Execution ===================== #
if __name__ == '__main__':
    imagePath = hough1Path

    image = cv2.imread(imagePath)
    image.astype('uint8')

    display('original', image)

    print('Computing...')

    edges = cv2.Canny(image, 50, 200)

    houghSpace, lines = hough(edges, 100)
    display('Hough Space', houghSpace)
    display('Hough', displayHoughLines(image, lines))

    print('Done!')

    cv2.waitKey(0)
