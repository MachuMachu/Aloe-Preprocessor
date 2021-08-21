import cv2 as cv, numpy as np


#resize image, input only one of h/w to preserve ratio
def resize(image, width = None, height = None, inter = cv.INTER_AREA):
    # initialize dimensions of img and get size
    dim = None
    (h, w) = image.shape[:2]

    # if both width and height are None, return the orig img
    if width is None and height is None:
        return image

    # check if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize
    resized = cv.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

##============================================================================
#for removing blurry background, not really optimal if bg is not really blurred
def remove(img):
    # grayscale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # canny
    canned = cv.Canny(gray, 70, 70);

    # dilate to close holes in lines
    kernel = np.ones((4,4),np.uint8)
    mask = cv.dilate(canned, kernel, iterations =3)

    # find contours
    contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # find big contours
    biggest_cntr = None;
    biggest_area = 0;
    for contour in contours:
        area = cv.contourArea(contour);
        if area > biggest_area:
            biggest_area = area;
            biggest_cntr = contour;

    # draw contours
    crop_mask = np.zeros_like(mask);
    cv.drawContours(crop_mask, [biggest_cntr], -1, (255), -1);

    # fill in holes
    # inverted
    inverted = cv.bitwise_not(crop_mask);

    # contours again
    contours, _ = cv.findContours(inverted, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE);

    # find small contours
    small_cntrs = [];
    for contour in contours:
        area = cv.contourArea(contour);
        if area < 200:
            print(area);
            small_cntrs.append(contour);

    # draw on mask
    cv.drawContours(crop_mask, small_cntrs, -1, (255), -1);

    # opening + median blur to smooth jaggies
    crop_mask = cv.erode(crop_mask, kernel, iterations = 1);
    crop_mask = cv.dilate(crop_mask, np.ones((6,6),np.uint8), iterations = 10);
    crop_mask = cv.medianBlur(crop_mask, 5);

    # crop image
    crop = np.zeros_like(img);
    crop[crop_mask == 255] = img[crop_mask == 255];

    return crop

#====================================================================================

##Main preprocessor to 
def preprocessor(img, stretch="auto"):
    contr =img.copy()
    #convert image to hsv color space so light intensity is of minimal factor
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    

    #thresholding color with respect to range
    minimum = np.array([40,40,40], np.uint8)
    maximum = np.array([70,255,255], np.uint8)
    thres = cv.inRange(hsv, minimum, maximum)
    
    #find contours within the thresholded
    contours, heirarchy = cv.findContours(thres, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(contr, contours, -1, (0,0,255), 2)
    
    #getting the Area of the outermost contours
    areas = [cv.contourArea(c) for c in contours]
    max_index = np.argmax(areas)
    cnt = contours[max_index]

    #getting and setting perimeters from the Area
    x,y,w,h = cv.boundingRect(cnt)
    cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    #cv.rectangle(hsv,(x,y),(x+w,y+h),(0,255,0),2)

    if((y+h)<100 or (x+w)<100):
        print("Aloe not detected")
        cropped = np.zeros((255,255,3), dtype=np.uint8)
        cv.putText(cropped,"Empty", (50,50),cv.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,255), 1, 1)
    else:
        #image segmentation/cropping
        cropped = hsv[y:y+h,x:x+w]


    #create empty image
    blank = np.zeros((255,255,3), dtype=np.uint8)
    
    if stretch == "auto":
        if cropped.shape[1]>cropped.shape[0]:
            cropped = resize(cropped, width=255)
        else:
            cropped = resize(cropped, height=255)

    cv.imshow("crop",cropped)
##    cropped = rmv.remove(cropped)
##
##    cv.imshow("Removed bg",cropped)
    
    off_x = (256 - cropped.shape[1])//2
    off_y = (256 - cropped.shape[0])//2
    blank [off_y:off_y+cropped.shape[0], off_x:off_x+cropped.shape[1]] = cropped
    retr_img=blank

    return retr_img             


##An agressive version of above, should delete left over bg from the -
##cropped image, although it removes too much
def preprocessor_Aggressive(img, stretch="auto"):
    contr =img.copy()
    #convert image to hsv color space so light intensity is of minimal factor
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    

    #thresholding color with respect to range
    minimum = np.array([40,40,30], np.uint8)
    maximum = np.array([70,255,255], np.uint8)
    thres = cv.inRange(hsv, minimum, maximum)
    
    #find contours within the thresholded
    contours, heirarchy = cv.findContours(thres, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(contr, contours, -1, (0,0,255), 2)
    
    #getting the Area of the outermost contours
##    areas = [cv.contourArea(c) for c in contours]
    areas = [cv.contourArea(c) for c in contours]
    max_index = np.argmax(areas)
    cnt = contours[max_index]

    #getting and setting perimeters from the Area
    x,y,w,h = cv.boundingRect(cnt)
    cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    #cv.rectangle(hsv,(x,y),(x+w,y+h),(0,255,0),2)


    if((y+h)<100 or (x+w)<100):
        print("Aloe not detected")
        cropped = np.zeros((255,255,3), dtype=np.uint8)
        cv.putText(cropped,"Empty", (50,50),cv.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,255), 3, -1)
    else:
        #image segmentation/cropping
        cropped = hsv[y:y+h,x:x+w]

    #create empty image
    blank = np.zeros((255,255,3), dtype=np.uint8)
    
    if stretch == "auto":
        if cropped.shape[1]>cropped.shape[0]:
            cropped = resize(cropped, width=255)
        else:
            cropped = resize(cropped, height=255)

    
    try:
        cropped = remove(cropped)
    except Exception as e: print(e)
    cv.imshow("removed",cropped)

    off_x = (256 - cropped.shape[1])//2
    off_y = (256 - cropped.shape[0])//2
    blank [off_y:off_y+cropped.shape[0], off_x:off_x+cropped.shape[1]] = cropped
    retr_img=blank

    return retr_img     


