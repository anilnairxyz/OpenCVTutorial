```
import cv2
from cv2 import cv
import numpy as np
from matplotlib import pyplot as plt
```
**Reading, Displaying and Writing an image**
```
img_file = "image.jpg"
cv2.namedWindow("example", cv2.WINDOW_NORMAL)
img = cv2.imread(img_file, 0)
cv2.imshow("example", img)
cv2.imwrite("image.jpg", img)
cv2.waitKey(0)
cv2.destroyWindow("example")
```
**Displaying image using matplotlib**
```
plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()
cv2.waitKey(0)
```
**Displaying video**
```
cv2.namedWindow("example", cv2.WINDOW_NORMAL)
vid_file = "video.mp4"
vid = cv2.VideoCapture(vid_file)
# vid = cv2.VideoCapture(0) # from webcam
spf = int(1000/vid.get(cv.CV_CAP_PROP_FPS))
while(True):
    # Capture frame-by-frame
    ret, vid_frame = vid.read()
    # vid_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Display the resulting frame
    cv2.imshow("example", vid_frame)
    if cv2.waitKey(spf) == ord('q'):
        break
    # When everything done, release the capture
vid.release()
cv2.destroyAllWindows()
```
**Writing video**
```
cv2.namedWindow("example", cv2.WINDOW_NORMAL)
vid = cv2.VideoCapture(0) # from webcam
fourcc = cv2.cv.FOURCC('D', 'I', 'V', 'X')
output_file = "video.avi"
fps = 33
fr_size = (640,480)
out = cv2.VideoWriter(output_file, fourcc, fps, fr_size)
while(vid.isOpened()):
    ret, frame = vid.read()
    if ret==True:
        frame = cv2.flip(frame,0)
        # write the flipped frame
        out.write(frame)
        cv2.imshow('example',frame)
        if cv2.waitKey(fps) == ord('q'):
            break
    else:
        break
vid.release()
out.release()
cv2.destroyAllWindows()
```
**Ceating borders on images**
```
img_file = "image.jpg"
cv2.namedWindow("example", cv2.WINDOW_NORMAL)
img = cv2.imread(img_file)
img[51:250, 51:250] = [0, 255, 0]  # green
img[251:450, 51:250] = [255, 0, 0] # blue
img[451:650, 51:250] = [0, 0, 255] # red
img[:,:,2] = 0 # remove red
img[:,:,0] = 0 # remove blue
BLUE = [255,0,0]
replicate = cv2.copyMakeBorder(img,100,100,100,100,cv2.BORDER_REPLICATE)
reflect = cv2.copyMakeBorder(img,100,100,100,100,cv2.BORDER_REFLECT)
reflect101 = cv2.copyMakeBorder(img,100,100,100,100,cv2.BORDER_REFLECT_101)
wrap = cv2.copyMakeBorder(img,100,100,100,100,cv2.BORDER_WRAP)
constant= cv2.copyMakeBorder(img,100,100,100,100,cv2.BORDER_CONSTANT,value=BLUE)
cv2.imshow("example", img)
cv2.waitKey(0)
cv2.imshow("example", replicate)
cv2.waitKey(0)
cv2.imshow("example", reflect)
cv2.waitKey(0)
cv2.imshow("example", reflect101)
cv2.waitKey(0)
cv2.imshow("example", wrap)
cv2.waitKey(0)
cv2.imshow("example", constant)
cv2.waitKey(0)
cv2.destroyWindow("example")
```
**Mixing images (using weights)**
```
img_file = "image.jpg"
cv2.namedWindow("example", cv2.WINDOW_NORMAL)
img = cv2.imread(img_file)
img_file2 = "logo.png"
img2 = cv2.imread(img_file2)
img_mixed = cv2.addWeighted(img,0.7,img2,0.3,0)
cv2.imshow('example',img_mixed)
cv2.waitKey(0)
cv2.destroyAllWindows()
```
**Mixing images (adding logo to corner of image)**
```
img_file = "image.jpg"
cv2.namedWindow("example", cv2.WINDOW_NORMAL)
img = cv2.imread(img_file)
img_file2 = "logo.png"
img2 = cv2.imread(img_file2)
I want to put logo on top-left corner, So I create a ROI
rows,cols,channels = img2.shape
roi = img[0:rows, 0:cols ]
Now create a mask of logo and create its inverse mask also
img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(img2gray, 150, 255, cv2.THRESH_BINARY)
mask_inv = cv2.bitwise_not(mask)
Now black-out the area of logo in ROI
img_bg = cv2.bitwise_and(roi,roi,mask = mask)
Take only region of logo from logo image.
img2_fg = cv2.bitwise_and(img2,img2,mask = mask_inv)
Put logo in ROI and modify the main image
img_mixed = cv2.add(img_bg,img2_fg)
img[0:rows, 0:cols ] = img_mixed
cv2.imshow('example',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```
**Image Processing - color tracking**
```
vid_file = "test_blue.avi"
vid = cv2.VideoCapture(vid_file)
cv2.namedWindow("example", cv2.WINDOW_NORMAL)
cv2.namedWindow("mask", cv2.WINDOW_NORMAL)
cv2.namedWindow("res", cv2.WINDOW_NORMAL)
while(1):
    # Take each frame
    ret, frame = vid.read()
    # Convert BGR to HSV
    # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # define range of blue color in HSV
    lower_blue = np.array([150,0,0])
    upper_blue = np.array([255,120,120])
    lower_yellow = np.array([0,150,150])
    upper_yellow = np.array([120,255,255])
    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(frame, lower_blue, upper_blue)
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame,frame, mask= mask)
    cv2.imshow('example',frame)
    cv2.imshow('mask',mask)
    cv2.imshow('res',res)
    if cv2.waitKey(33) == ord('q'):
        break
vid.release()
cv2.waitKey(0)
cv2.destroyAllWindows()
```
**Image Processing - Adaptive thresholding (getting outlines - sketches)**
```
img_file = "image.jpg"
img = cv2.imread(img_file,0)
img = cv2.medianBlur(img,5)
ret,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,11,2)
th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)
titles = ['Original Image', 'Global Thresholding (v = 127)',
            'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
images = [img, th1, th2, th3]
for i in xrange(4):
    plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()
```
**Image Processing - Canny edge detection (outlines - sketches)**
```
img_file = "image.jpg"
img = cv2.imread(img_file,0)
edges = cv2.Canny(img,25,75)
plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()
```
**Image Processing - Histogram backprojection (matching histograms)**
```
img_file = "image.jpg"
roi = cv2.imread(img_file)
hsv = cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
# target is the image we search in
tgt_file = "target.jpg"
target = cv2.imread(tgt_file)
hsvt = cv2.cvtColor(target,cv2.COLOR_BGR2HSV)
# calculating object histogram
roihist = cv2.calcHist([hsv],[0, 1], None, [180, 256], [0, 180, 0, 256] )
# normalize histogram and apply backprojection
cv2.normalize(roihist,roihist,0,255,cv2.NORM_MINMAX)
dst = cv2.calcBackProject([hsvt],[0,1],roihist,[0,180,0,256],1)
# Now convolute with circular disc
disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
cv2.filter2D(dst,-1,disc,dst)
# threshold and binary AND
ret,thresh = cv2.threshold(dst,50,255,0)
thresh = cv2.merge((thresh,thresh,thresh))
res = cv2.bitwise_and(target,thresh)
res = np.vstack((target,thresh,res))
# test
cv2.namedWindow("example", cv2.WINDOW_NORMAL)
cv2.imshow('example',res)
cv2.waitKey(0)
cv2.destroyAllWindows()
```
**Image Processing - Fourier Transform**
```
img_file = "image.jpg"
img = cv2.imread(img_file,0)
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20*np.log(np.abs(fshift))
plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()
```
**Image Processing - Template Matching**
```
img_file = "image.jpg"
img = cv2.imread(img_file,0)
img2 = img.copy()
template_file = "template.jpg"
template = cv2.imread(template_file,0)
w, h = template.shape[::-1]
# All the 6 methods for comparison in a list
# methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
#            'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
method = eval('cv2.TM_SQDIFF')
# Apply template Matching
res = cv2.matchTemplate(img,template,method)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
# If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
    top_left = min_loc
else:
    top_left = max_loc
bottom_right = (top_left[0] + w, top_left[1] + h)
cv2.rectangle(img,top_left, bottom_right, 255, 2)
# test
cv2.namedWindow("example", cv2.WINDOW_NORMAL)
cv2.imshow('example',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```
**Image Processing - Template Matching (using thresholds)**
```
img_file = "image.jpg"
img = cv2.imread(img_file,0)
img_rgb = cv2.imread(img_file)
template_file = "template.jpg"
template = cv2.imread(template_file,0)
w, h = template.shape[::-1]
method = eval('cv2.TM_SQDIFF_NORMED')
# Apply template Matching
res = cv2.matchTemplate(img,template,method)
threshold = 0.35
loc = np.where(res<=threshold)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
print min_val
for pt in zip(*loc[::-1]):
    cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,255,0), 2)
# test
cv2.namedWindow("example", cv2.WINDOW_NORMAL)
cv2.imshow('example',img_rgb)
cv2.waitKey(0)
cv2.destroyAllWindows()
```
**Video - Template Matching**
```
cv2.namedWindow("example", cv2.WINDOW_NORMAL)
vid_file = "video.mp4"
vid = cv2.VideoCapture(vid_file)
# vid = cv2.VideoCapture(0) # from webcam
spf = int(1000/vid.get(cv.CV_CAP_PROP_FPS))
template_file = "template.png"
template = cv2.imread(template_file,0)
w, h = template.shape[::-1]
method = eval('cv2.TM_SQDIFF')
while(vid.isOpened()):
    # Capture frame-by-frame
    ret, vid_frame = vid.read()
    frame_img = cv2.cvtColor(vid_frame, cv2.COLOR_BGR2GRAY)
    res = cv2.matchTemplate(frame_img,template,method)
    threshold = 50000000 
    loc = np.where(res<=threshold)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    for pt in zip(*loc[::-1]):
	cv2.rectangle(vid_frame, pt, (pt[0] + w, pt[1] + h), (0,255,0), 2)
    # Display the resulting frame
    cv2.imshow("example", vid_frame)
    if cv2.waitKey(spf) == ord('q'):
	break
    # When everything done, release the capture
vid.release()
cv2.destroyAllWindows()
```
**OCR from image - Using pytesser**
```
from pytesser import *
from PIL import Image
img_file = "image.jpg"
img = Image.open(img_file)
print image_to_string(img)
cv2.waitKey(0)
```
**GaussianBlur, pyrDown (downsizing+Blur),**
```
img_file = "image.jpg"
cv2.namedWindow("example", cv2.WINDOW_NORMAL)
img = cv2.imread(img_file)
res = cv2.GaussianBlur(img, (5, 5), 0, 0)
res = cv2.pyrDown(img)
cv2.imshow("example", res)
cv2.waitKey(0)
cv2.destroyWindow("example")
```
**Contour finding and drawing**
```
cv2.namedWindow("example", cv2.WINDOW_NORMAL)
img_file = "logo.png"
img = cv2.imread(img_file)
imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(imgray,200,255,0)
contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img, contours, -1, (0,255,0), 3)
cv2.imshow("example", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```
**Object Tracking using MOG / MOG2**
```
cv2.namedWindow("video", cv2.WINDOW_NORMAL)
cv2.namedWindow("ObjectTracking", cv2.WINDOW_NORMAL)
vid_file = "video.avi"
vid = cv2.VideoCapture(vid_file)
fps = 30
output_file = "video_out.avi"
fourcc = cv2.cv.FOURCC('D', 'I', 'V', 'X')
fr_size = (640,480)
out = cv2.VideoWriter(output_file, fourcc, fps, fr_size)
# bgmask = cv2.BackgroundSubtractorMOG(300, 6, 0.6, 1)
bgmask = cv2.BackgroundSubtractorMOG2(300, 16, True)
while(vid.isOpened()):
    # Capture frame-by-frame
    ret, vid_frame = vid.read()
    ret, vid_frame_copy = vid.read()
    # Background subtract
    fgmask = bgmask.apply(vid_frame)
    # Further processing
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (12, 12))
    fgerode = cv2.erode(fgmask, kernel)
    fgdilate = cv2.dilate(fgerode, kernel)
    contours, hier = cv2.findContours(fgdilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(vid_frame, contours, -1, (0, 0, 255), 3)
    # Display the resulting frame
    cv2.imshow("video", vid_frame_copy)
    cv2.imshow("ObjectTracking", vid_frame)
    out.write(vid_frame)
    if cv2.waitKey(6) == ord('q'):
        break
    # When everything done, release the capture
vid.release()
cv2.destroyAllWindows()
```
**Grabbing frames from a video**
```
cv2.namedWindow("example", cv2.WINDOW_NORMAL)
vid_file = "video.avi"
vid = cv2.VideoCapture(vid_file)
spf = int(1000/vid.get(cv.CV_CAP_PROP_FPS))
grabno = 0
while(vid.isOpened()):
    # Capture frame-by-frame
    ret, vid_frame = vid.read()
    # Display the resulting frame
    cv2.imshow("example", vid_frame)
    img_file = "grabimg_"+format(grabno)+".jpg"
    keyin = cv2.waitKey(spf)
    if keyin == ord('q'):
        break
    # When everything done, release the capture
    elif keyin == ord('g'):
        cv2.imwrite(img_file, vid_frame)
	grabno += 1
    # Grab the frame
vid.release()
cv2.destroyAllWindows()
```
**Accurate Template Matching for video frames**
```
cv2.namedWindow("example", cv2.WINDOW_NORMAL)
vid_file = "video.avi"
vid = cv2.VideoCapture(vid_file)
spf = int(1000/vid.get(cv.CV_CAP_PROP_FPS))
target_no = [1, 2, 3, 4, 5, 6, 7]
colour_code = [(255,255,255), (255,0,0), (0,255,0), (0,0,255), (0,0,0), (255,255,0), (255,0,255)]
while(vid.isOpened()):
     # Capture frame-by-frame
     ret, vid_frame = vid.read()
     # Display the resulting frame
     for img_no in target_no: 
          template_file = "grabimg_"+format(img_no)+".jpg"
          template = cv2.imread(template_file,0)
          frame_img = cv2.cvtColor(vid_frame, cv2.COLOR_BGR2GRAY)
          method = eval('cv2.TM_CCOEFF_NORMED')
          # Apply template Matching
          res = cv2.matchTemplate(frame_img,template,method)
          threshold = 0.90
          loc = np.where(res>=threshold)
          for pt in zip(*loc[::-1]):
              cv2.rectangle(vid_frame, pt, (pt[0] + 100, pt[1] + 100), colour_code[int(img_no)-1], 10)
              cv2.waitKey(0)
     cv2.imshow("example", vid_frame)
     keyin = cv2.waitKey(1)
     if keyin == ord('q'):
         break
# test
cv2.waitKey(0)
cv2.destroyAllWindows()
```
**Template Matching for video frames - with downsampling / Gaussian filter**
```
cv2.namedWindow("example", cv2.WINDOW_NORMAL)
vid_file = "video.avi"
vid = cv2.VideoCapture(vid_file)
spf = int(1000/vid.get(cv.CV_CAP_PROP_FPS))
target_no = [1, 2, 3, 4, 5, 6, 7]
colour_code = [(255,255,255), (255,0,0), (0,255,0), (0,0,255), (0,0,0), (255,255,0), (255,0,255)]
while(vid.isOpened()):
     # Capture frame-by-frame
     ret, vid_frame = vid.read()
     # Display the resulting frame
     for img_no in target_no: 
          template_file = "grabimg_"+format(img_no)+".jpg"
          template = cv2.imread(template_file,0)
          template = cv2.GaussianBlur(template, (5, 5), 0, 0)
          template = cv2.resize(template,(0,0), 0.5, 0.5, cv.CV_INTER_AREA)
          frame_img = cv2.cvtColor(vid_frame, cv2.COLOR_BGR2GRAY)
          frame_img = cv2.GaussianBlur(frame_img, (5, 5), 0, 0)
          frame_img = cv2.resize(frame_img,(0,0), 0.5, 0.5, cv.CV_INTER_AREA)
          method = eval('cv2.TM_CCOEFF_NORMED')
          # Apply template Matching
          res = cv2.matchTemplate(frame_img,template,method)
          threshold = 0.999
          loc = np.where(res>=threshold)
          for pt in zip(*loc[::-1]):
              cv2.rectangle(vid_frame, pt, (pt[0] + 100, pt[1] + 100), colour_code[int(img_no)-1], 10)
              cv2.waitKey(0)
     cv2.imshow("example", vid_frame)
     keyin = cv2.waitKey(1)
     if keyin == ord('q'):
         break
# test
cv2.waitKey(0)
cv2.destroyAllWindows()
```
**Template Matching for video frames - using Histogram comparison**
```
cv2.namedWindow("example", cv2.WINDOW_NORMAL)
vid_file = "video.avi"
vid = cv2.VideoCapture(vid_file)
spf = int(1000/vid.get(cv.CV_CAP_PROP_FPS))
target_no = [1, 2, 3, 4, 5, 6, 7]
colour_code = [(255,255,255), (255,0,0), (0,255,0), (0,0,255), (0,0,0), (255,255,0), (255,0,255)]
while(vid.isOpened()):
     # Capture frame-by-frame
     ret, vid_frame = vid.read()
     # Display the resulting frame
     for img_no in target_no: 
          template_file = "grabimg_"+format(img_no)+".jpg"
          template = cv2.imread(template_file)
          template_hsv = cv2.cvtColor(template, cv2.COLOR_BGR2HSV)
          frame_hsv = cv2.cvtColor(vid_frame, cv2.COLOR_BGR2HSV)
          # calculating object histograms
          frame_hist = cv2.calcHist([frame_hsv],[0, 1], None, [180, 256], [0, 180, 0, 256] )
          template_hist = cv2.calcHist([template_hsv],[0, 1], None, [180, 256], [0, 180, 0, 256] )
          # normalize histogram
          cv2.normalize(frame_hist,frame_hist,0,255,cv2.NORM_MINMAX)
          cv2.normalize(template_hist,template_hist,0,255,cv2.NORM_MINMAX)
          # Compare Histograms
          res = cv2.compareHist(frame_hist,template_hist,cv.CV_COMP_BHATTACHARYYA)
	  threshold = 0.1
	  if res < threshold:
               cv2.rectangle(vid_frame, pt, (pt[0] + 100, pt[1] + 100), colour_code[int(img_no)-1], 10)
               cv2.waitKey(0)
     cv2.imshow("example", vid_frame)
     keyin = cv2.waitKey(1)
     if keyin == ord('q'):
         break
# test
cv2.waitKey(0)
cv2.destroyAllWindows()
```
**Displaying multiple video streams**
```
cv2.namedWindow("example1", cv2.WINDOW_NORMAL)
cv2.namedWindow("example2", cv2.WINDOW_NORMAL)
vid1 = cv2.VideoCapture(0) # from inbuilt webcam
vid2 = cv2.VideoCapture(1) # from inbuilt webcam
while(vid1.isOpened() and vid2.isOpened()):
    # Capture frame-by-frame
    ret, vid1_frame = vid1.read()
    ret, vid2_frame = vid2.read()
    # Display the resulting frame
    cv2.imshow("example1", vid1_frame)
    cv2.imshow("example2", vid2_frame)
    if cv2.waitKey(33) == ord('q'):
        break
    # When everything done, release the capture
vid1.release()
vid2.release()
cv2.destroyAllWindows()
```
**Writing video from TV tuner card**
```
cv2.namedWindow("example", cv2.WINDOW_NORMAL)
vid = cv2.VideoCapture(1)
fourcc = cv2.cv.FOURCC('D', 'I', 'V', 'X')
output_file = "C:/Users/Anil/Desktop/write-vid.avi"
frame_width = int(vid.get(cv.CV_CAP_PROP_FRAME_WIDTH))
print frame_width
frame_height = int(vid.get(cv.CV_CAP_PROP_FRAME_HEIGHT))
print frame_height
fr_size = (frame_width, frame_height)
out = cv2.VideoWriter(output_file, fourcc, 30, fr_size)
while(vid.isOpened()):
    ret, frame = vid.read()
    # write the frame
    if ret == True:
        out.write(frame)
        cv2.imshow('example',frame)
    if cv2.waitKey(1) == ord('q'):
        break
vid.release()
out.release()
cv2.destroyAllWindows()
```
**Cropping videos**
```
cv2.namedWindow("example", cv2.WINDOW_NORMAL)
input_file = "video.avi"
vid = cv2.VideoCapture(input_file)
fourcc = int(vid.get(cv.CV_CAP_PROP_FOURCC))
fps = int(vid.get(cv.CV_CAP_PROP_FPS))
output_file = "video_out.avi"
frame_width = int(vid.get(cv.CV_CAP_PROP_FRAME_WIDTH))
frame_height = int(vid.get(cv.CV_CAP_PROP_FRAME_HEIGHT))
fr_size = (frame_width, frame_height)
out = cv2.VideoWriter(output_file, fourcc, fps, fr_size)
grab = False
while(vid.isOpened()):
    ret, frame = vid.read()
    # write the frame
    if ret == True:
	if grab == True:
            out.write(frame)
        cv2.imshow('example',frame)
    keyin = cv2.waitKey(1)
    if keyin == ord('q'):
        break
    elif keyin == ord('g'):
	grab = True
	print "Grab"
    elif keyin == ord('h'):
	grab = False
	print "Hold"
vid.release()
out.release()
cv2.destroyAllWindows()
```
**Template matching : Multiple Commercials**
<br />
Full template, unoptimized (10min / video min for 20 frames) 
```
cv2.namedWindow("example", cv2.WINDOW_NORMAL)
vid_file = "video.avi"
vid = cv2.VideoCapture(vid_file)
log_file = "log.txt"
log = open(log_file, 'w')
ad1_no = [0, 1, 2, 3, 4, 5, 6]
ad2_no = [7, 8, 9, 10, 11, 12, 13, 14, 15]
ad3_no = [16, 17, 18, 19]
ad_nos = [0]*20
prev_ad_no = 20
while(vid.isOpened()):
     # Capture frame-by-frame
     ret, vid_frame = vid.read()
     frame_img = cv2.cvtColor(vid_frame, cv2.COLOR_BGR2GRAY)
     method = eval('cv2.TM_CCOEFF_NORMED')
     threshold = 0.98
     # Display the resulting frame
     for img_no in ad1_no: 
          template_file = "grabimg_"+format(img_no)+".jpg"
          template = cv2.imread(template_file,0)
          # Apply template Matching
          res = cv2.matchTemplate(frame_img,template,method)
          loc = np.where(res>=threshold)
          for pt in zip(*loc[::-1]):
               if prev_ad_no != img_no:
                    ad_nos[int(img_no)]+=1
                    print "ad1_"+format(img_no)+"="+format(ad_nos[int(img_no)])+";"
                    log.write("ad1_"+format(img_no)+"="+format(ad_nos[int(img_no)])+"\n")
		    prev_ad_no = img_no
     for img_no in ad2_no: 
          template_file = "grabimg_"+format(img_no)+".jpg"
          template = cv2.imread(template_file,0)
          # Apply template Matching
          res = cv2.matchTemplate(frame_img,template,method)
          loc = np.where(res>=threshold)
          for pt in zip(*loc[::-1]):
               if prev_ad_no != img_no:
                    ad_nos[int(img_no)]+=1
                    print "ad2_"+format(img_no)+"="+format(ad_nos[int(img_no)])+";"
                    log.write("ad2_"+format(img_no)+"="+format(ad_nos[int(img_no)])+"\n")
		    prev_ad_no = img_no
     for img_no in ad3_no: 
          template_file = "grabimg_"+format(img_no)+".jpg"
          template = cv2.imread(template_file,0)
          # Apply template Matching
          res = cv2.matchTemplate(frame_img,template,method)
          loc = np.where(res>=threshold)
          for pt in zip(*loc[::-1]):
               if prev_ad_no != img_no:
                    ad_nos[int(img_no)]+=1
                    print "ad3_"+format(img_no)+"="+format(ad_nos[int(img_no)])+";"
                    log.write("ad3_"+format(img_no)+"="+format(ad_nos[int(img_no)])+"\n")
		    prev_ad_no = img_no
     cv2.imshow("example", vid_frame)
     keyin = cv2.waitKey(1)
     if keyin == ord('q'):
         break
# test
log.close()
cv2.waitKey(0)
cv2.destroyAllWindows()
```
**Template matching : Multiple Commercials**
<br />
5 sub-templates, optimized (1.5 min / video min for 20 frames) 
```
cv2.namedWindow("example", cv2.WINDOW_NORMAL)
vid_file = "video.avi"
vid = cv2.VideoCapture(vid_file)
log_file = "log.txt"
log = open(log_file, 'w')
ad1_no = [0, 1, 2, 3, 4, 5, 6]
ad2_no = [7, 8, 9, 10, 11, 12, 13, 14, 15]
ad3_no = [16, 17, 18, 19]
ad_nos = [0]*20
prev_ad_no = 20
template_register = []
iy = int(480)
ix = int(720)
for img_no in ad1_no + ad2_no + ad3_no: 
     template_array = []
     adimages_file = "grabimg_"+format(img_no)+".jpg"
     ad_img = cv2.imread(adimages_file,0)
     template_array.append(ad_img[100:200,100:200])
     template_array.append(ad_img[(iy-200):(iy-100),100:200])
     template_array.append(ad_img[(iy/2-50):(iy/2+50),(ix/2-50):(ix/2+50)])
     template_array.append(ad_img[100:200,(ix-200):(ix-100)])
     template_array.append(ad_img[(iy-200):(iy-100),(ix-200):(ix-100)])
     template_register.append(np.concatenate(template_array, 0))
method = eval('cv2.TM_CCOEFF_NORMED')
threshold = 0.95
while(vid.isOpened()):
     # Capture frame-by-frame
     ret, vid_frame = vid.read()
     frame_img = cv2.cvtColor(vid_frame, cv2.COLOR_BGR2GRAY)
     frame_array = []
     frame_array.append(frame_img[100:200,100:200])
     frame_array.append(frame_img[(iy-200):(iy-100),100:200])
     frame_array.append(frame_img[(iy/2-50):(iy/2+50),(ix/2-50):(ix/2+50)])
     frame_array.append(frame_img[100:200,(ix-200):(ix-100)])
     frame_array.append(frame_img[(iy-200):(iy-100),(ix-200):(ix-100)])
     frame_img = np.concatenate(frame_array,0)
     # Display the resulting frame
     for img_no in ad1_no + ad2_no + ad3_no: 
          # Apply template Matching
          res = cv2.matchTemplate(frame_img,template_register[int(img_no)],method)
          loc = np.where(res>=threshold)
          for pt in zip(*loc[::-1]):
               if prev_ad_no != img_no:
                    ad_nos[int(img_no)]+=1
                    print "adframe_"+format(img_no)+"="+format(ad_nos[int(img_no)])+";"
                    log.write("adframe_"+format(img_no)+"="+format(ad_nos[int(img_no)])+"\n")
                    prev_ad_no = img_no
     cv2.imshow("example", vid_frame)
     keyin = cv2.waitKey(1)
     if keyin == ord('q'):
         break
# test
log.close()
cv2.waitKey(0)
cv2.destroyAllWindows()
```
**Template matching : Multiple Commercials**
<br />
5 sub-templates, aggregate video frames before template match 
(1.25 min / video min for 20 frames) 
```
cv2.namedWindow("example", cv2.WINDOW_NORMAL)
vid_file = "video.avi"
vid = cv2.VideoCapture(vid_file)
log_file = "log.txt"
log = open(log_file, 'w')
ad1_no = [0, 1, 2, 3, 4, 5, 6]
ad2_no = [7, 8, 9, 10, 11, 12, 13, 14, 15]
ad3_no = [16, 17, 18, 19]
ad_nos = [0]*20
prev_ad_no = 20
template_register = []
iy = int(480)
ix = int(720)
for img_no in ad1_no + ad2_no + ad3_no: 
     template_array = []
     adimages_file = "grabimg_"+format(img_no)+".jpg"
     ad_img = cv2.imread(adimages_file,0)
     template_array.append(ad_img[100:200,100:200])
     template_array.append(ad_img[(iy-200):(iy-100),100:200])
     template_array.append(ad_img[(iy/2-50):(iy/2+50),(ix/2-50):(ix/2+50)])
     template_array.append(ad_img[100:200,(ix-200):(ix-100)])
     template_array.append(ad_img[(iy-200):(iy-100),(ix-200):(ix-100)])
     template_register.append(np.concatenate(template_array, 0))
method = eval('cv2.TM_CCOEFF_NORMED')
threshold = 0.95
while(vid.isOpened()):
     frame_register = []
     # Capture frame-by-frame
     for i in range(90):
          ret, vid_frame = vid.read()
          frame_img = cv2.cvtColor(vid_frame, cv2.COLOR_BGR2GRAY)
          frame_array = []
          frame_array.append(frame_img[100:200,100:200])
          frame_array.append(frame_img[(iy-200):(iy-100),100:200])
          frame_array.append(frame_img[(iy/2-50):(iy/2+50),(ix/2-50):(ix/2+50)])
          frame_array.append(frame_img[100:200,(ix-200):(ix-100)])
          frame_array.append(frame_img[(iy-200):(iy-100),(ix-200):(ix-100)])
          frame_img = np.concatenate(frame_array,0)
	  frame_register.append(frame_img)
     frame_img = np.concatenate(frame_register,0)
     # Display the resulting frame
     for img_no in ad1_no + ad2_no + ad3_no: 
          # Apply template Matching
          res = cv2.matchTemplate(frame_img,template_register[int(img_no)],method)
          loc = np.where(res>=threshold)
          for pt in zip(*loc[::-1]):
               if prev_ad_no != img_no:
                    ad_nos[int(img_no)]+=1
                    print "adframe_"+format(img_no)+"="+format(ad_nos[int(img_no)])+";"
                    log.write("adframe_"+format(img_no)+"="+format(ad_nos[int(img_no)])+"\n")
                    prev_ad_no = img_no
     cv2.imshow("example", vid_frame)
     keyin = cv2.waitKey(1)
     if keyin == ord('q'):
         break
# test
log.close()
cv2.waitKey(0)
cv2.destroyAllWindows()
```
**Template matching : Multiple Commercials**
<br />
templates / frames sliced (1/100), 
aggregate video frames before template match 
(9.5s / video min for 20 frames) 
```
cv2.namedWindow("example", cv2.WINDOW_NORMAL)
vid_file = "video.avi"
vid = cv2.VideoCapture(vid_file)
log_file = "log.txt"
log = open(log_file, 'w')
ad1_no = [0, 1, 2, 3, 4, 5, 6]
ad2_no = [7, 8, 9, 10, 11, 12, 13, 14, 15]
ad3_no = [16, 17, 18, 19]
ad_nos = [0]*20
prev_ad_no = 20
template_register = []
iy = int(480)
ix = int(720)
for img_no in ad1_no + ad2_no + ad3_no: 
     adimages_file = "grabimg_"+format(img_no)+".jpg"
     ad_img = cv2.imread(adimages_file,0)
     template_img = ad_img[::10, ::10]
     template_register.append(template_img)
method = eval('cv2.TM_CCORR_NORMED')
threshold = 0.98
while(vid.isOpened()):
     frame_register = []
     # Capture frame-by-frame
     for i in range(90):
          ret, vid_frame = vid.read()
          frame_grey = cv2.cvtColor(vid_frame, cv2.COLOR_BGR2GRAY)
	  frame_img = frame_grey[::10, ::10]
	  frame_register.append(frame_img)
     frame_img = np.concatenate(frame_register,0)
     # Display the resulting frame
     for img_no in ad1_no + ad2_no + ad3_no: 
          # Apply template Matching
          res = cv2.matchTemplate(frame_img,template_register[int(img_no)],method)
          loc = np.where(res>=threshold)
          for pt in zip(*loc[::-1]):
               if prev_ad_no != img_no:
                    ad_nos[int(img_no)]+=1
                    print "adframe_"+format(img_no)+"="+format(ad_nos[int(img_no)])+";"
                    log.write("adframe_"+format(img_no)+"="+format(ad_nos[int(img_no)])+"\n")
                    prev_ad_no = img_no
     cv2.imshow("example", vid_frame)
     keyin = cv2.waitKey(1)
     if keyin == ord('q'):
         break
# test
log.close()
cv2.waitKey(0)
cv2.destroyAllWindows()
```
**Template matching : Multiple Commercials**
<br />
Histogram comparison
(not working satisfactorily)
```
cv2.namedWindow("example", cv2.WINDOW_NORMAL)
vid_file = "video.avi"
vid = cv2.VideoCapture(vid_file)
log_file = "log.txt"
log = open(log_file, 'w')
ad1_no = [0, 1, 2, 3, 4, 5, 6]
ad2_no = [7, 8, 9, 10, 11, 12, 13, 14, 15]
ad3_no = [16, 17, 18, 19]
ad_nos = [0]*20
prev_ad_no = 20
template_register = []
iy = int(480)
ix = int(720)
for img_no in ad1_no + ad2_no + ad3_no: 
     adimages_file = "grabimg_"+format(img_no)+".jpg"
     ad_img = cv2.imread(adimages_file)
     adimg_hsv = cv2.cvtColor(ad_img,cv2.COLOR_BGR2HSV)
     adimg_hist = cv2.calcHist([adimg_hsv],[0, 1], None, [180, 256], [0, 180, 0, 256] )
     cv2.normalize(adimg_hist,adimg_hist,0,255,cv2.NORM_MINMAX)
     template_register.append(adimg_hist)
method = eval('cv.CV_COMP_CORREL')
threshold = 0.99
while(vid.isOpened()):
     # Capture frame-by-frame
     ret, vid_frame = vid.read()
     frame_hsv = cv2.cvtColor(vid_frame, cv2.COLOR_BGR2HSV)
     frame_hist = cv2.calcHist([frame_hsv],[0, 1], None, [180, 256], [0, 180, 0, 256] )
     cv2.normalize(frame_hist,frame_hist,0,255,cv2.NORM_MINMAX)
     # Display the resulting frame
     for img_no in ad1_no + ad2_no + ad3_no: 
          # Apply template Matching
          res = cv2.compareHist(frame_hist,template_register[int(img_no)],method)
	  if (res>=threshold):
               if prev_ad_no != img_no:
                    ad_nos[int(img_no)]+=1
                    print "adframe_"+format(img_no)+"="+format(ad_nos[int(img_no)])+";"
                    log.write("adframe_"+format(img_no)+"="+format(ad_nos[int(img_no)])+"\n")
                    prev_ad_no = img_no
     cv2.imshow("example", vid_frame)
     keyin = cv2.waitKey(1)
     if keyin == ord('q'):
         break
# test
log.close()
cv2.waitKey(0)
cv2.destroyAllWindows()
```
**Converting videos into image banks**
```
cv2.namedWindow("example", cv2.WINDOW_NORMAL)
input_file = "video.avi"
vid = cv2.VideoCapture(input_file)
img_no = 0
while(vid.isOpened()):
    ret, frame = vid.read()
    # write the frame
    img_file = "img_"+format(img_no)+".jpg"
    cv2.imwrite(img_file, frame)
    cv2.imshow('example',frame)
    img_no += 1
    keyin = cv2.waitKey(1)
    if keyin == ord('q'):
        break
vid.release()
cv2.destroyAllWindows()
