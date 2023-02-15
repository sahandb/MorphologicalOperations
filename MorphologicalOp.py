import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import copy

matplotlib.rcParams['figure.figsize'] = (6.0, 6.0)
matplotlib.rcParams['image.cmap'] = 'gray'


def dilate(im, element):
    kernelSize = element.shape[0]
    height, width = im.shape[:2]

    border = kernelSize // 2
    # Threshold image

    if len(im.shape) > 2:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    ret, im = cv2.threshold(im, 127, 255, cv2.THRESH_BINARY)

    # Create a padded image with zeros paddedIm
    paddedIm = np.zeros((height + border * 2, width + border * 2))
    paddedIm = cv2.copyMakeBorder(im, border, border, border, border, cv2.BORDER_CONSTANT, value=0)
    for h_i in range(border, height + border):
        for w_i in range(border, width + border):
            # When you find a white pixel
            if im[h_i - border, w_i - border] == 255:
                paddedIm[h_i - border: (h_i + border) + 1, w_i - border: (w_i + border) + 1] = 255
    return paddedIm[border:height + border, border:width + border]


def erode(im, element):
    kernelSize = element.shape[0]
    height, width = im.shape[:2]

    border = kernelSize // 2

    # Threshold image
    if len(im.shape) > 2:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    ret, im = cv2.threshold(im, 127, 255, cv2.THRESH_BINARY)

    # Create a padded image with zeros paddedIm
    paddedIm = np.full((height + border * 2, width + border * 2), 255)
    paddedIm = cv2.copyMakeBorder(im, border, border, border, border, cv2.BORDER_CONSTANT, value=255)
    for h_i in range(border, height + border):
        for w_i in range(border, width + border):
            # When you find a black pixel
            if im[h_i - border, w_i - border] == 0:
                paddedIm[h_i - border: (h_i + border) + 1, w_i - border: (w_i + border) + 1] = cv2.bitwise_and(
                    paddedIm[h_i - border: (h_i + border) + 1, w_i - border: (w_i + border) + 1],
                    element)
    return paddedIm[border:height + border, border:width + border]


def extract(im):
    height, width = im.shape[:2]
    border = 1
    paddedIm = np.full((height + border * 2, width + border * 2), 255)
    paddedIm = cv2.copyMakeBorder(im, border, border, border, border, cv2.BORDER_CONSTANT, value=255)

    components = {}
    counter = 1
    for h_i in range(height):
        for w_i in range(width):
            if paddedIm[h_i + 1, w_i + 1] != 0:
                movingKernel = np.array(
                    [paddedIm[h_i + 1, w_i], paddedIm[h_i, w_i], paddedIm[h_i, w_i + 1], paddedIm[h_i, w_i + 2]])
                if paddedIm[h_i + 1, w_i] == 0 and paddedIm[h_i, w_i] == 0 and paddedIm[h_i, w_i + 1] == 0 and paddedIm[
                    h_i, w_i + 2] == 0:
                    components[counter] = counter
                    paddedIm[h_i + border, w_i + border] = counter
                    counter += 1
                elif np.count_nonzero(movingKernel) == 1:
                    paddedIm[h_i + border, w_i + border] = np.sum(movingKernel)
                elif np.count_nonzero(movingKernel) > 1:
                    paddedIm[h_i + border, w_i + border] = np.min(movingKernel[np.nonzero(movingKernel)])

                    if paddedIm[h_i + 1, w_i] != 0:
                        components[paddedIm[h_i + 1, w_i]] = np.min(movingKernel[np.nonzero(movingKernel)])
                    if paddedIm[h_i, w_i] != 0:
                        components[paddedIm[h_i, w_i]] = np.min(movingKernel[np.nonzero(movingKernel)])
                    if paddedIm[h_i, w_i + 1] != 0:
                        components[paddedIm[h_i, w_i + 1]] = np.min(movingKernel[np.nonzero(movingKernel)])
                    if paddedIm[h_i, w_i + 2] != 0:
                        components[paddedIm[h_i, w_i + 2]] = np.min(movingKernel[np.nonzero(movingKernel)])

    for k in components.keys():
        components[k] = components[components[k]]

    components[0] = 0
    for h_i in range(paddedIm.shape[0]):
        for w_i in range(paddedIm.shape[1]):
            paddedIm[h_i, w_i] = components[paddedIm[h_i, w_i]]

    setsOfClass = set(paddedIm.flat)
    c = 0
    for sett in setsOfClass:
        c = c + 1
        abcd = copy.deepcopy(paddedIm)
        for i in range(height):
            for j in range(width):
                if abcd[i][j] != sett:
                    abcd[i][j] = 0
        ddd = np.logical_and(abcd, paddedIm)
        plt.imshow(ddd)
        plt.show()
    return len(setsOfClass)


# global
cross_structure = np.array([[1, 0, 1],
                            [0, 1, 0],
                            [1, 0, 1]], dtype=np.uint8)

diamond_structure = np.array([[0, 1, 0],
                              [1, 1, 1],
                              [0, 1, 0]], dtype=np.uint8)


def partA():
    # Read the input image
    imageName = "./img1.tif"
    image = cv2.imread(imageName)

    # Apply dilate function on the input image
    imageDilated_diamond = dilate(image, diamond_structure)
    imageDilated_cross = dilate(image, cross_structure)

    plt.figure(figsize=[15, 5])
    plt.subplot(131)
    plt.imshow(image)
    plt.title("Original Image")
    plt.subplot(132)
    plt.imshow(imageDilated_diamond)
    plt.title("Dilated Image Diamond")
    plt.subplot(133)
    plt.imshow(imageDilated_cross)
    plt.title("Dilated Image Cross")
    plt.show()
    # Eroding the image , decreases brightness of image
    imageEroded_diamond = erode(image, diamond_structure)
    imageEroded_cross = erode(image, cross_structure)

    plt.figure(figsize=[15, 5])
    plt.subplot(131)
    plt.imshow(image)
    plt.title("Original Image")
    plt.subplot(132)
    plt.imshow(imageEroded_diamond)
    plt.title("Eroded Image Diamond")
    plt.subplot(133)
    plt.imshow(imageEroded_cross)
    plt.title("Eroded Image Cross")
    plt.show()


def partB():
    # Read the input image
    imageName = "./img2.tif"
    image = cv2.imread(imageName)
    imageEroded_diamond = erode(image, diamond_structure)
    boundary_image = image[:, :, 0] - imageEroded_diamond

    plt.figure(figsize=[15, 15])
    plt.subplot(121)
    plt.imshow(image)
    plt.title("Original Image")
    plt.subplot(122)
    plt.imshow(boundary_image)
    plt.title("boundary image")
    plt.show()


def partC():
    # Read the input image
    imageName = "./img3.tif"
    image = cv2.imread(imageName)
    structure_element1 = np.ones((11, 11), dtype=np.uint8)
    imageEroded_11 = erode(image, structure_element1)

    numberOfClass = extract(imageEroded_11)
    print(numberOfClass)


def main():
    partA()
    # partB()
    # partC()


if __name__ == '__main__':
    main()
