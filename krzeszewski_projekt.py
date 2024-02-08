import numpy as np
import cv2

wideo = cv2.VideoCapture('PW_SW_9.avi')

while(wideo.isOpened()):
    udany_odczyt, ramka = wideo.read()
    if udany_odczyt:  # udany odczyt ramki
        
        
        hsv = cv2.cvtColor(ramka, cv2.COLOR_BGR2HSV)
        lower_country = np.array([12, 20, 200])
        upper_country = np.array([25, 255, 255]) 
        mask = cv2.inRange(hsv, lower_country, upper_country)
        cv2.imshow('ramka',ramka)
        cv2.imshow('maska',mask)
        # cv2.waitKey(10)   # jedna klatka na 33ms = 30 fps
        cv2.waitKey(0)   # czekamy na wcisniecie klawisz po kazdej klatce
    else:   # koniec pliku
        wideo.release()            
        cv2.waitKey(1) 
        cv2.destroyAllWindows()
        cv2.waitKey(1)
