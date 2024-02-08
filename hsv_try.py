import cv2
import numpy as np

def show(image,title=""):
    # if obraz.ndim == 2:
        # cv2.imshow(obraz,cmap='gray')
    # else:
    cv2.namedWindow(title, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.imshow(title,image)
    # plt.title(tytul)   

#turqoise mask function
def turqoise_mask(input_image):
    hsv = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)
    turqoise_lower = np.array([80, 0, 0])
    turqoise_upper = np.array([110, 255, 255])  
    mask_turqoise = cv2.inRange(hsv, turqoise_lower, turqoise_upper)
    return mask_turqoise

obraz_we = cv2.imread('PW_SW_9_ref.png') 


#2. Wykonaj segmentację koloru obrazu wejściowego, w wyniku której powstanie obraz binarny zawierający wszystkie obszary mapy o kolorze takim jak kolor jakim zaznaczono województwo, które masz wyodrębnić z obrazu wejściowego (województwo referencyjne)-> 5 pkt
mask = turqoise_mask(obraz_we)

#117 175 104
show(obraz_we,"1. bazowy obraz")
show(mask,"2. mask")
cv2.waitKey(0)