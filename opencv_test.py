import cv2 as cv

img = cv.imread("chuck_norris.jpg")

cv.imshow("Display window" , img)

k = cv.waitKey(0)
