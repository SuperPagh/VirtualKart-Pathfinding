import cv2
import math
from cv2 import cv
import numpy as np
from scipy.misc import imread
from scipy.linalg import norm
from scipy import sum, average

vidcap = cv2.VideoCapture('videoplayback.mp4')
success, image = vidcap.read()
count = 0
success = True

def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err

lastImages = []
#http://docs.opencv.org/3.2.0/d4/dc6/tutorial_py_template_matching.html
position = [0, 0]
angle = 1.57
angles = [angle]
positions = []
positions.append([position[0], position[1]])
lengths = []
deltaAngels = []

while success:
    success,image = vidcap.read()
    if not success:
        break
    image = cv2.resize(image, (0,0), fx=.5, fy=.5)
    #print 'Read a new frame: ', success
    cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file
    #lastImages.append(image)
    if len(lastImages) > 2 and False:
        lastImages.pop(0)
        if count % 1 == 0:
            print count
            height, width, channels = image.shape
            bestMSE = 100000000
            bestScale = 0
            bestPosX = 0
            method = cv.CV_TM_SQDIFF_NORMED

            c = 0.0
            for scale in range(10000, 9940, -2): # range(1, .69, -.05)
                scale /= 10000.0;
                scaled_img = cv2.resize(image, (0,0), fx=scale, fy=scale)
                small_image = scaled_img
                large_image = lastImages[0].copy()
                result = cv2.matchTemplate(small_image, large_image, method)
                mn,_,mnLoc,_ = cv2.minMaxLoc(result)
                MPx,MPy = mnLoc
                trows,tcols = small_image.shape[:2]
                #cv2.rectangle(large_image, (MPx,MPy),(MPx+tcols,MPy+trows),(0,0,255),2)
                #cv2.imshow('output',large_image)
                #cv2.waitKey(0)
                midX = (MPx + tcols / 2.0) / width
                midY = (MPy + trows / 2.0) / height
                #print str(midX) + " " + str(midY)
                #print np.amax(result)
                error = np.amax(result)

                bestPosX = bestPosX * c + midX
                c += 1.0
                bestPosX /= c

                if error < bestMSE:
                    bestMSE = error
                    bestScale = scale

            if (bestPosX - .5) > .005:
                angle -= (bestPosX - .5) * 2 # just some factor
            position[0] += math.cos(angle) * (1 - bestScale)
            position[1] += math.sin(angle) * (1 - bestScale)
            positions.append([position[0], position[1]])
            angles.append(angle)
            lengths.append(1 - bestScale)
            deltaAngels.append(bestPosX - .5)

                
    count += 1

for p in positions:
    print p

print angles
print lengths
print deltaAngels





"""for posX in range(0, int((1 - scale) * 1000) + 10, 10):
                posX /= 1000.0
                for posY in range(0, int((1 - scale) * 1000) + 10, 10):
                    posY /= 1000.0
                    scaled_img = cv2.resize(image, (0,0), fx=scale, fy=scale)
                    scaled_height, scaled_width, scaled_channels = scaled_img.shape 
                    yOffset = int(posY * height)
                    xOffset = int(posX * width)
                    crop_img = lastImages[0][yOffset:yOffset + scaled_height, xOffset:xOffset + scaled_width]
                    error = compare_images(scaled_img, crop_img)[1]
                    if error < bestMSE:
                        bestMSE = error
                        bestScale = scale
                        bestPosX = posX
                        bestPosY = posY
                        i1 = scaled_img
                        i2 = crop_img



        #oldImage = cv2.resize(image, (0,0), fx=0.5, fy=0.5) 
        #crop_img = img[200:400, 100:300]"""
