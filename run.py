from preprocess import pre
import cv2 as cv


# READ IMAGE
x="1.png"
image = cv.imread("images/"+str(x))
image = pre.resize(image, height=255)
image2 = image.copy()
#check if the image is loaded
cv.imshow("Image", image)

image = pre.preprocessor_Aggressive(image)
#cv.imshow("return", image)
cv.imshow("Aggressive",cv.cvtColor(image,cv.COLOR_HSV2BGR))
image2 = pre.preprocessor(image2)
cv.imshow("Non-Aggressive",cv.cvtColor(image2,cv.COLOR_HSV2BGR))

cv.imwrite("images/out/"+x+"-hsv.png",image)
cv.imwrite("images/out/"+x+"-rgb.png",cv.cvtColor(image,cv.COLOR_HSV2BGR))
cv.imwrite("images/out/"+x+"-hsv2.png",image2)
cv.imwrite("images/out/"+x+"-rgb2.png",cv.cvtColor(image2,cv.COLOR_HSV2BGR))

##======For Testing================
##thres_img, contr_img, hsv_img, bound_img,crop_img = pr.preprocessor_test_(image)
##cv.imshow("thresholded", thres_img)
##cv.imshow("Contours", contr_img)
##cv.imshow("HSV", hsv_img)
##cv.imshow("Bounds", bound_img)
##cv.imshow("Cropped", crop_img)
##=================================



####Press any key to exit###
cv.waitKey(0)    
cv.destroyAllWindows()
