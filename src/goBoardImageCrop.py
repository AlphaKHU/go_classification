import imutils
import cv2
import datetime
import numpy as np
import os

inputImagePath = os.path.abspath("./src/inputImage/input.png")
outputImagePath = os.path.abspath("./src/outputImage/")
image = cv2.imread(inputImagePath)
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image_gray_blurred = cv2.GaussianBlur(image_gray, (5, 5), 0)

ret,thresh1 = cv2.threshold(image_gray_blurred,127,255,cv2.THRESH_BINARY)
ret,thresh2 = cv2.threshold(image_gray_blurred,127,255,cv2.THRESH_BINARY_INV)
ret,thresh3 = cv2.threshold(image_gray_blurred,127,255,cv2.THRESH_TRUNC)
ret,thresh4 = cv2.threshold(image_gray_blurred,127,255,cv2.THRESH_TOZERO)
ret,thresh5 = cv2.threshold(image_gray_blurred,127,255,cv2.THRESH_TOZERO_INV)

images_row1 = np.hstack([image_gray_blurred, thresh1, thresh2])
images_row2 = np.hstack([thresh3, thresh4, thresh5])
images_combined = np.vstack((images_row1, images_row2))


cv2.imshow('Images', images_combined)
cv2.waitKey(0)


class ShapeDetector:
    def __init__(self):
        pass

    def detect(self, c):
	# initialize the shape name and approximate the contour
        shape = "unidentified"
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
		# if the shape is a triangle, it will have 3 vertices
        if len(approx) == 3:
            shape = "triangle"

		# if the shape has 4 vertices, it is either a square or
		# a rectangle
        elif len(approx) == 4:
			# compute the bounding box of the contour and use the
			# bounding box to compute the aspect ratio
            (x, y, w, h) = cv2.boundingRect(approx)
            ar = w / float(h)

			# a square will have an aspect ratio that is approximately
			# equal to one, otherwise, the shape is a rectangle
            shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"

		# if the shape is a pentagon, it will have 5 vertices
        elif len(approx) == 5:
            shape = "pentagon"

		# otherwise, we assume the shape is a circle
        else:
            shape = "circle"
		# return the name of the shape
        return shape

sd = ShapeDetector()

# Find contours in the image.
cnts, hierarchy = cv2.findContours(thresh4.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)


for c in cnts:
    if sd.detect(c) != 'rectangle': next
    c = c.astype("float")
    c = c.astype("int")
    x, y, w, h = cv2.boundingRect(c)
    print(w)
    print(h)
    
    if h < 700 or w < 700:
        continue

    now = datetime.datetime.now().strftime("%d_%H-%M-%S")
    cv2.imwrite(str(outputImagePath) + str(now) + ".png", image[y: y + h, x: x + w])
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow("image", image)
    cv2.waitKey(0)

cv2.destroyAllWindows()