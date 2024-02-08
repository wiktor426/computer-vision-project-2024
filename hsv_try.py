import cv2
import numpy as np

def pokaz(obraz,tytul=""):
    # if obraz.ndim == 2:
        # cv2.imshow(obraz,cmap='gray')
    # else:
    cv2.namedWindow(tytul, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.imshow(tytul,obraz)
    # plt.title(tytul)   

obraz_we = cv2.imread('PW_SW_9_ref.png') 
pokaz(obraz_we,"1. bazowy obraz")

#2. Wykonaj segmentację koloru obrazu wejściowego, w wyniku której powstanie obraz binarny zawierający wszystkie obszary mapy o kolorze takim jak kolor jakim zaznaczono województwo, które masz wyodrębnić z obrazu wejściowego (województwo referencyjne)-> 5 pkt
hsv = cv2.cvtColor(obraz_we, cv2.COLOR_BGR2HSV)
#117 175 104
turqoise_lower = np.array([80, 0, 0])
turqoise_upper = np.array([110, 255, 255])  
mask_turqoise = cv2.inRange(hsv, turqoise_lower, turqoise_upper)
pokaz(mask_turqoise,"2. mask")
cv2.waitKey(0)