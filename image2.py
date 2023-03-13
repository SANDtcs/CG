import matplotlib.pyplot as plt
import cv2
import numpy as np




# Colour Convolution
# BGR, GRAY, RGB, HSV
img = cv2.imread('/content/Fox.jpeg')
 
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV);
 
cv2_imshow(hsv)
     
# cv2.imwrite('blur_kernel.jpg', img)





# THRESHOLDING

img = cv2.imread('/content/Fox.jpeg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ret, thresh = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)
#ret, thresh = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY_INV)
#ret, thresh = cv2.threshold(img, 120, 255, cv2.THRESH_TRUNC)
#ret, thresh = cv2.threshold(img, 120, 255, cv2.THRESH_TOZERO)
#ret, thresh = cv2.threshold(img, 120, 255, cv2.THRESH_TOZERO_INV)
  
cv2_imshow(thresh)


# HISTOGRAM ANALYSIS

img = cv2.imread('/content/Fox.jpeg', 0)

histr = cv2.calcHist([img], [0], None, [256], [0,256])

plt.plot(histr)
plt.show()



# HISTOGRAM EQUALIZATION(STRETCHING)

img = cv2.imread('/content/Fox.jpeg', 0)

equ = cv2.equalizeHist(img)
# cv2.imwrite('foxEqu.jpeg', equ)

res = np.hstack((img, equ))

cv2_imshow(res)

histr = cv2.calcHist([equ], [0], None, [256], [0,256])

plt.plot(histr)
plt.show()


# In[ ]:


# HISTOGRAM MATCHING
from skimage import exposure
from skimage.exposure import match_histograms

img = cv2.imread('/content/Fox.jpeg', 0)
ref = cv2.imread('/content/foxEqu.jpeg', 0)

matched = match_histograms(img, ref, multichannel=True)

res = np.hstack((img, matched))

cv2_imshow(res)


# In[ ]:


# INTENSITY TRANSFORMATION

#Log Transformations
img = cv2.imread('/content/Fox.jpeg')

c = 255/(np.log(1 + np.max(img)))
log_transformed = c * np.log(1 + img)
  
log_transformed = np.array(log_transformed, dtype = np.uint8)
  
cv2_imshow(log_transformed)


# In[ ]:


# Power-Law (Gamma) Transformation
img = cv2.imread('/content/Fox.jpeg')

gamma = [0.1, 0.5, 1.2, 2.2]

gamma_corrected = np.array(255*(img / 255) ** gamma[3], dtype = 'uint8')
  
cv2_imshow(gamma_corrected)


# In[ ]:


# Piecewise-Linear Transformation Functions

def pixelVal(pix, r1, s1, r2, s2):
    if (0 <= pix and pix <= r1):
        return (s1 / r1)*pix
    elif (r1 < pix and pix <= r2):
        return ((s2 - s1)/(r2 - r1)) * (pix - r1) + s1
    else:
        return ((255 - s2)/(255 - r2)) * (pix - r2) + s2

img = cv2.imread('/content/Fox.jpeg')

r1 = 70
s1 = 0
r2 = 140
s2 = 255
  
pixelVal_vec = np.vectorize(pixelVal)
  
contrast_stretched = pixelVal_vec(img, r1, s1, r2, s2)
  
cv2_imshow(contrast_stretched)


# In[ ]:


# EROSION

img = cv2.imread('/content/Fox.jpeg', 0)

binr = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
kernel = np.ones((5, 5), np.uint8)
invert = cv2.bitwise_not(binr)
erosion = cv2.erode(invert, kernel, iterations = 1)

plt.imshow(erosion, cmap = 'gray')


# In[ ]:


# DILATION

img = cv2.imread('/content/Fox.jpeg', 0)

binr = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
kernel = np.ones((3, 3), np.uint8)
invert = cv2.bitwise_not(binr)
dilation = cv2.dilate(invert, kernel, iterations = 1)

plt.imshow(dilation, cmap = 'gray')


# In[ ]:


# OPENING

img = cv2.imread('/content/Fox.jpeg', 0)

binr = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(binr, cv2.MORPH_OPEN, kernel, iterations = 1)

plt.imshow(opening, cmap = 'gray')


# In[ ]:


# CLOSING

img = cv2.imread('/content/Fox.jpeg', 0)

binr = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
kernel = np.ones((3, 3), np.uint8)
closing = cv2.morphologyEx(binr, cv2.MORPH_CLOSE, kernel, iterations=1)

plt.imshow(closing, cmap = 'gray')


# In[ ]:


# HIT-OR-MISS TRANSFORMATION

img = cv2.imread('/content/Fox.jpeg', 0)

kernel = np.array((
        [1, 1, 1],
        [0, 1, -1],
        [0, 1, -1]), dtype="int")
 
output_image = cv2.morphologyEx(img, cv2.MORPH_HITMISS, kernel)

plt.imshow(output_image, cmap = 'gray')


# In[ ]:


# SOBEL EDGE DETECTION

img = cv2.imread('/content/Fox.jpeg')
size = 9

imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray, (size, size), 0)

# Sobel Edge Detection
sobelX = cv2.Sobel(src=imgBlur, ddepth = cv2.CV_64F, dx = 1, dy = 0, ksize = 5)
sobelY = cv2.Sobel(src=imgBlur, ddepth = cv2.CV_64F, dx = 0, dy = 1, ksize = 5)
sobelXY = cv2.Sobel(src=imgBlur, ddepth = cv2.CV_64F, dx = 1, dy = 1, ksize = 5)

plt.figure(figsize = (11, 6))

plt.subplot(141)
plt.imshow(img, cmap = 'gray')
plt.title('Original')
plt.xticks([])
plt.yticks([])

plt.subplot(142)
plt.imshow(sobelX, cmap = 'gray')
plt.title('Sobel X')
plt.xticks([])
plt.yticks([])

plt.subplot(143)
plt.imshow(sobelY, cmap = 'gray')
plt.title('Sobel Y')
plt.xticks([])
plt.yticks([])

plt.subplot(144)
plt.imshow(sobelXY, cmap = 'gray')
plt.title('Sobel XY')
plt.xticks([])
plt.yticks([])

plt.show()


# In[ ]:


# CANNY EDGE DETECTION

img = cv2.imread('/content/Fox.jpeg')
size = 9

imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
imgBlur = cv2.GaussianBlur(imgGray, (size, size), 0)

# Sobel Edge Detection
canny = cv2.Canny(image=imgBlur, threshold1 = 100, threshold2 = 200)

plt.figure(figsize = (11, 6))

plt.subplot(121)
plt.imshow(img, cmap = 'gray')
plt.title('Original')
plt.xticks([])
plt.yticks([])

plt.subplot(122)
plt.imshow(canny, cmap = 'gray')
plt.title('Canny Edge Detection')
plt.xticks([])
plt.yticks([])

plt.show()


# In[ ]:


# BLUR

img = cv2.imread('/content/Fox.jpeg', cv2.IMREAD_UNCHANGED)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB);

averageBlur = cv2.blur(img, (5, 5))
  
plt.imshow(averageBlur)


# In[ ]:


# BLUR(Convolution)

img = cv2.imread('/content/Fox.jpeg', cv2.IMREAD_UNCHANGED)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB);

kernel2 = np.ones((5, 5), np.float32) / 25

img = cv2.filter2D(src=img, ddepth=-1, kernel=kernel2)

plt.imshow(img)


# In[ ]:


# GAUSSIAN BLUR

img = cv2.imread('/content/Fox.jpeg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB);

gaussian = cv2.GaussianBlur(img, (3, 3), 0)

plt.imshow(gaussian)


# In[ ]:


# MEDIAN BLUR

img = cv2.imread('/content/Fox.jpeg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB);

medianBlur = cv2.medianBlur(img, 9)

plt.imshow(medianBlur)


# In[ ]:


# BILATERAL BLUR

img = cv2.imread('/content/Fox.jpeg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB);

bilateral = cv2.bilateralFilter(img, 9, 75, 75)

plt.imshow(bilateral)


# In[ ]:


# SHARPEN(Convolution)

img = cv2.imread('/content/Fox.jpeg', cv2.IMREAD_UNCHANGED)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB);

kernel2 = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])

img = cv2.filter2D(src=img, ddepth=-1, kernel=kernel2)

plt.imshow(img)


# In[ ]:


# FREQUENCY FILTERS
# DOMAIN FILTER

img = cv2.imread('/content/Fox.jpeg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB);

domainFilter = cv2.edgePreservingFilter(img, flags=1, sigma_s=60, sigma_r=0.6)

plt.imshow(domainFilter)


# In[ ]:


# MEAN FILTER

img = cv2.imread('/content/Fox.jpeg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB);

kernel = np.ones((10,10),np.float32)/25
meanFilter = cv2.filter2D(img,-1,kernel)

plt.imshow(meanFilter)


# In[ ]:


# MEDIAN FILTER

img = cv2.imread('/content/Fox.jpeg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB);

kernel = np.ones((10,10),np.float32)/25
meanFilter = cv2.filter2D(img,-1,kernel)

plt.imshow(meanFilter)


# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------

# In[ ]:


from PIL import Image
from IPython.display import display


# In[ ]:


try:
    img = Image.open('/content/Images/Colours.jpg')
except IOError:
    print("ERROR in Opening Image")
    pass

width, height = img.size

display(img)


# **Histogram Equalization for Low Constrast Images**

# In[ ]:


import matplotlib.pyplot as plt
from google.colab.patches import cv2_imshow
import cv2
import numpy as np


# In[ ]:


bush = cv2.imread('/content/Images/Bush.jpg')
histOld = cv2.calcHist([bush], [0], None, [256], [0, 256])
plt.plot(histOld)

cv2_imshow(bush)
plt.show()


# In[ ]:


equ = cv2.equalizeHist(bush)
histNew = cv2.calcHist([equ], [0], None, [256], [0, 256])
plt.plot(histNew)

res = np.hstack((bush, equ))
cv2_imshow(res)


# **Filters**

# **Mean**

# In[ ]:


import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image, ImageFilter


# In[ ]:


img = cv2.imread('/content/Images/Nadal.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
size = 9
newImg = cv2.blur(img, (size, size))

plt.figure(figsize = (11, 6))

plt.subplot(121)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_HSV2RGB))
plt.title('Original')
plt.xticks([])
plt.yticks([])

plt.subplot(122)
plt.imshow(cv2.cvtColor(newImg, cv2.COLOR_HSV2RGB))
plt.title('Mean Filter')
plt.xticks([])
plt.yticks([])

plt.show()


# In[ ]:


img = cv2.imread('/content/Images/Nadal.jpg')
img2 = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
size = 9

newImg = cv2.blur(img2, (size, size))

plt.figure(figsize = (11, 6))

plt.subplot(121)
plt.imshow(img2, cmap = 'gray')
plt.title('Original')

plt.subplot(122)
plt.imshow(newImg, cmap = 'gray')
plt.title('Mean Filter')

plt.show()


# **Gaussian Filter**

# In[ ]:


img = cv2.imread('/content/Images/Nadal.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
size = 9
newImg = cv2.GaussianBlur(img, (size, size), 0)

plt.figure(figsize = (11, 6))

plt.subplot(121)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_HSV2RGB))
plt.title('Original')
plt.xticks([])
plt.yticks([])

plt.subplot(122)
plt.imshow(cv2.cvtColor(newImg, cv2.COLOR_HSV2RGB))
plt.title('Gaussian Filter')
plt.xticks([])
plt.yticks([])

plt.show()


# In[ ]:


img = cv2.imread('/content/Images/Nadal.jpg')
img2 = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
size = 9

newImg = cv2.GaussianBlur(img2, (size, size), 0)

plt.figure(figsize = (11, 6))

plt.subplot(121)
plt.imshow(img2, cmap = 'gray')
plt.title('Original')

plt.subplot(122)
plt.imshow(newImg, cmap = 'gray')
plt.title('Gaussian Filter')

plt.show()


# **Median**

# In[ ]:


img = cv2.imread('/content/Images/Nadal.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
size = 9
newImg = cv2.medianBlur(img, size)

plt.figure(figsize = (11, 6))

plt.subplot(121)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_HSV2RGB))
plt.title('Original')
plt.xticks([])
plt.yticks([])

plt.subplot(122)
plt.imshow(cv2.cvtColor(newImg, cv2.COLOR_HSV2RGB))
plt.title('Median Filter')
plt.xticks([])
plt.yticks([])

plt.show()


# In[ ]:


img = cv2.imread('/content/Images/Nadal.jpg')
img2 = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
size = 9

newImg = cv2.medianBlur(img2, size)

plt.figure(figsize = (11, 6))

plt.subplot(121)
plt.imshow(img2, cmap = 'gray')
plt.title('Original')

plt.subplot(122)
plt.imshow(newImg, cmap = 'gray')
plt.title('Median Filter')

plt.show()


# **Laplacian Filter**

# In[ ]:


img = cv2.imread('/content/Images/Nadal.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
size = 9

newImg = cv2.Laplacian(img, cv2.CV_64F, ksize = 3)

plt.figure(figsize = (11, 6))

plt.subplot(121)
plt.imshow(img, cmap = 'gray')
plt.title('Original')
plt.xticks([])
plt.yticks([])

plt.subplot(122)
plt.imshow(img + newImg, cmap = 'gray')
plt.title('Result')
plt.xticks([])
plt.yticks([])

plt.show()


# **Custom Filter**

# In[ ]:


img = cv2.imread('/content/Images/Nadal.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
size = 9

kernel = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
newImg = cv2.filter2D(img, -1, kernel)

plt.figure(figsize = (11, 6))

plt.subplot(121)
plt.imshow(img)
plt.title('Original')
plt.xticks([])
plt.yticks([])

plt.subplot(122)
plt.imshow(newImg)
plt.title('Result')
plt.xticks([])
plt.yticks([])

plt.show()


# **Edge Detection**

# Sobel Edge Detection

# In[ ]:


img = cv2.imread('/content/Images/Nadal.jpg')
size = 9

imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray, (size, size), 0)

# Sobel Edge Detection
sobelX = cv2.Sobel(src=imgBlur, ddepth = cv2.CV_64F, dx = 1, dy = 0, ksize = 5)
sobelY = cv2.Sobel(src=imgBlur, ddepth = cv2.CV_64F, dx = 0, dy = 1, ksize = 5)
sobelXY = cv2.Sobel(src=imgBlur, ddepth = cv2.CV_64F, dx = 1, dy = 1, ksize = 5)

plt.figure(figsize = (11, 6))

plt.subplot(141)
plt.imshow(img, cmap = 'gray')
plt.title('Original')
plt.xticks([])
plt.yticks([])

plt.subplot(142)
plt.imshow(sobelX, cmap = 'gray')
plt.title('Sobel X')
plt.xticks([])
plt.yticks([])

plt.subplot(143)
plt.imshow(sobelY, cmap = 'gray')
plt.title('Sobel Y')
plt.xticks([])
plt.yticks([])

plt.subplot(144)
plt.imshow(sobelXY, cmap = 'gray')
plt.title('Sobel XY')
plt.xticks([])
plt.yticks([])


plt.show()


# Canny Edge Detection

# In[ ]:


img = cv2.imread('/content/Images/Nadal.jpg', cv2.IMREAD_UNCHANGED)
size = 9

imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
imgBlur = cv2.GaussianBlur(imgGray, (size, size), 0)

# Sobel Edge Detection
canny = cv2.Canny(image=imgBlur, threshold1 = 100, threshold2 = 200)

plt.figure(figsize = (11, 6))

plt.subplot(121)
plt.imshow(img, cmap = 'gray')
plt.title('Original')
plt.xticks([])
plt.yticks([])

plt.subplot(122)
plt.imshow(canny, cmap = 'gray')
plt.title('Canny Edge Detection')
plt.xticks([])
plt.yticks([])

plt.show()


# **Frequency Domain Filtering**

# In[ ]:


import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image, ImageFilter
from google.colab.patches import cv2_imshow

flower = cv2.imread('/content/Images/Flower.jpg', cv2.IMREAD_UNCHANGED)
cv2_imshow(flower)


# In[ ]:


flower = cv2.cvtColor(flower, cv2.COLOR_BGR2RGB)
domainFilter = cv2.edgePreservingFilter(flower, flags = 1, sigma_s = 60, sigma_r = 0.6)

plt.figure(figsize = (11, 6))

plt.subplot(121)
plt.imshow(flower)
plt.title('Original')
plt.xticks([])
plt.yticks([])

plt.subplot(122)
plt.imshow(domainFilter)
plt.title('Edge Preserving Filter')
plt.xticks([])
plt.yticks([])

plt.show()


# Bilateral Filter

# In[ ]:


flower = cv2.cvtColor(flower, cv2.COLOR_BGR2RGB)
bilateralFilter = cv2.bilateralFilter(flower, 60, 60, 60)

plt.figure(figsize = (11, 6))

plt.subplot(121)
plt.imshow(flower)
plt.title('Original')
plt.xticks([])
plt.yticks([])

plt.subplot(122)
plt.imshow(domainFilter)
plt.title('Bilateral Filter')
plt.xticks([])
plt.yticks([])

plt.show()


# **Morphological Operations**

# In[ ]:


import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('/content/Images/Alphabets.jpg', 0)

plt.imshow(img)


# Erosion

# In[ ]:


binr = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
kernel = np.ones((5, 5), np.uint8)
invert = cv2.bitwise_not(binr)
erosion = cv2.erode(invert, kernel, iterations = 1)

plt.imshow(erosion, cmap = 'gray')


# Dilation

# In[ ]:


binr = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
kernel = np.ones((3, 3), np.uint8)
invert = cv2.bitwise_not(binr)
dilation = cv2.dilate(invert, kernel, iterations = 1)

plt.imshow(dilation, cmap = 'gray')


# Opening

# In[ ]:


binr = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(binr, cv2.MORPH_OPEN, kernel, iterations = 1)

plt.imshow(opening, cmap = 'gray')


# Closing

# In[ ]:


binr = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
kernel = np.ones((3, 3), np.uint8)
closing = cv2.morphologyEx(binr, cv2.MORPH_OPEN, kernel, iterations = 1)

plt.imshow(closing, cmap = 'gray')

