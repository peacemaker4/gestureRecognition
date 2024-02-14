import handGesture as hg
import cv2

input = cv2.imread("images/image_001.png")
output = hg.handGesture(input)
cv2.imshow('', output)

cv2.waitKey(0)