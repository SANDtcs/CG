# ************** Sharperning *****************
import cv2
import numpy as np

def mean(image):
    mean_image= cv2.blur(image,(5,5))
    cv2.imshow('Mean filtered Image',mean_image)
    cv2.waitKey(0)
    
def median(image):
    median_image = cv2.medianBlur(image,5)
    cv2.imshow('Median filtered Image',median_image)
    cv2.waitKey(0)

def sharpening(image):
    kernel= np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
    sharpened_image =cv2.filter2D(image,-1,kernel)
    cv2.imshow('Sharpened Image',sharpened_image)
    cv2.waitKey(0)





if __name__ == '__main__':
    image = cv2.imread('tiger.jpg')
    mean(image)
    median(image)
    sharpening(image)
    
    
    
 
 
 
 # ***************** Morphological **********************
 
 def erosion(image):
    kernel=np.ones((5,5),np.uint8)
    eroded_image = cv2.erode(image,kernel,iterations=1)
    cv2.imshow('Eroded Image',eroded_image)
    cv2.waitKey(0)

def dilation(image):
    kernel =np.ones((5,5),np.uint8)
    dilated_image=cv2.dilate(image,kernel,iterations=1)
    cv2.imshow('Dilated Image',dilated_image)
    cv2.waitKey(0)

def opening(image):
    kernel =np.ones((5,5),np.uint8)
    opening_image=cv2.morphologyEx(image,cv2.MORPH_OPEN,kernel,iterations=1)
    cv2.imshow('Opening Image',opening_image)
    cv2.waitKey(0)

def closing(image):
    kernel=np.ones((5,5),np.uint8)
    closing_image=cv2.morphologyEx(image,cv2.MORPH_CLOSE,kernel,iterations=1)
    cv2.imshow('Closing Image',closing_image)
    cv2.waitKey(0)

def hitMiss(image):
    kernel=np.array(([1,1,1],[0,1,-1],[0,1,-1]),dtype = "int")
    grayImage=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    hitMissImage=cv2.morphologyEx(grayImage,cv2.MORPH_HITMISS,kernel,iterations=1)
    cv2.imshow('Hit Miss Image',hitMissImage)
    cv2.waitKey(0)


if __name__ == '__main__':
    erosion_image = cv2.imread('white.png')
    dilation_image =cv2.imread('black.png')
    erosion(erosion_image)
    dilation(dilation_image)
    opening(erosion_image)
    closing(dilation_image)
    hitMiss(erosion_image)
    hitMiss(dilation_image)
    
    
    
    
# ********************* histogram matching ***********************************
import cv2
from skimage.exposure import match_histograms

if __name__== '__main__':
    image = cv2.imread('flowerGray.jpg')
    reference=cv2.imread('equalizedFlower.jpg')
    matched = match_histograms(image, reference)
    cv2.imshow('Histogram matched image',matched)
    cv2.waitKey(0)
    
    
    

# ************************ Edge detection **********************************

import cv2


def sobel(image):
    grayImage = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    cv2.imshow('Gray Image',grayImage)
    cv2.waitKey(0)
    image_blur = cv2.GaussianBlur(grayImage,(3,3),0)
    after_sobel_image = cv2.Sobel(src=image_blur,ddepth = cv2.CV_64F,dx=1,dy=0,ksize=5)
    cv2.imshow('Sobel Image',after_sobel_image)
    cv2.waitKey(0)


def canny(image):
    grayImage = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image_blur = cv2.GaussianBlur(grayImage,(3,3),0)
    after_canny_image=cv2.Canny(image=image_blur,threshold1=100,threshold2=200)
    cv2.imshow('Canny Image',after_canny_image)
    cv2.waitKey(0)



if __name__=='__main__':
    image = cv2.imread('tiger.jpg')
    sobel(image)
    canny(image)
    



# ****************************** Histogram equaliztion *******************************************
import cv2
import matplotlib.pyplot as plt

def plotHist(image):
    grayImage = cv2.imread('flowerGray.jpg',0)
    histogram = cv2.calcHist([grayImage],[0],None,[256],[0,256])
    plt.plot(histogram)
    plt.show()
    

def histEq(image):
    grayImage = cv2.imread('flower.jpg',0)
    grayScaleImage = cv2.imread('flowerGray.jpg')
    cv2.imshow('Gray Scale Image', grayScaleImage)
    cv2.waitKey(0)
    equalized = cv2.equalizeHist(grayImage)
    cv2.imwrite('equalizedFlower.jpg', equalized)
    eqImage = cv2.imread('equalizedFlower.jpg',0)
    hist = cv2.calcHist(eqImage, [0], None,[256], [0,256])
    plt.plot(hist)
    plt.show()
    equalizedImage = cv2.imread('equalizedFlower.jpg')
    cv2.imshow('Equalized Image',equalizedImage)
    cv2.waitKey(0)


if __name__ == '__main__':
    image = cv2.imread('flower.jpg')
    cv2.imshow('Flower Image',image)
    cv2.waitKey(0)
    plotHist(image)
    histEq(image)
    



# ****************************** freqfilters ******************************
import cv2
import numpy as np

def lowPass(image):
    kernel = np.array([[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1]])
    img = cv2.filter2D(image,-1,kernel/sum(kernel))
    cv2.imshow('Low pass Image',img)
    cv2.waitKey(0)

def highPass(image):
    kernel =np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])
    img = cv2.filter2D(image,-1,kernel)
    cv2.imshow('High Pass Image',img)
    cv2.waitKey(0)
    kernel =np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    img_color =cv2.filter2D(image,-1,kernel)
    cv2.imshow('High Pass Image',img_color)
    cv2.waitKey(0)

if __name__=='__main__':
    image = cv2.imread('tiger.jpg')
    lowPass(image)
    highPass(image)
    
    

# *************************************** colour conversion ********************************
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#convert to black and white image
def blackAndWhite(image):
    grayImage = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    (threshold,binaryImage) = cv2.threshold(grayImage,125,255,cv2.THRESH_BINARY)
    cv2.imwrite('flowerBW.jpg',binaryImage)
    bw = cv2.imread('flowerBW.jpg')
    cv2.imshow('Black and White Image',bw)
    cv2.waitKey(0)


#convert to gray scale image 
def Gray(image):
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('flowerGray.jpg',grayImage)
    showImage = cv2.imread('flowerGray.jpg')
    cv2.imshow('GrayScale Image',showImage)
    cv2.waitKey(0)



if __name__ == '__main__':
    image_cv2 = cv2.imread('flower.jpg')
    cv2.imshow('Flower image',image_cv2)
    cv2.waitKey(0)
    Gray(image_cv2)
    blackAndWhite(image_cv2)
