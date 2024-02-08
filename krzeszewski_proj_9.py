import cv2
import numpy as np
import keyboard
import matplotlib.pyplot as plt
   

TURQOISE_COUNT = 0
RED_COUNT = 0
YELLOW_COUNT = 0
results_list = []

def show(image,title=""):
    # if obraz.ndim == 2:
        # cv2.imshow(obraz,cmap='gray')
    # else:
    cv2.namedWindow(title, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.imwrite(title + ".jpg", image) 
    cv2.imshow(title,image)
    # plt.title(tytul)   

#turqoise mask function
def turqoise_mask(input_image):
    hsv = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)
    turqoise_lower = np.array([80, 200, 160])
    turqoise_upper = np.array([110, 255, 255])  
    mask_turqoise = cv2.inRange(hsv, turqoise_lower, turqoise_upper)
    show(mask_turqoise,"mask_turqoise")
    return mask_turqoise

def red_mask(input_image):
    hsv = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)
    red_lower = np.array([160, 170, 20])
    red_upper = np.array([180, 255, 255])  
    mask_red = cv2.inRange(hsv, red_lower, red_upper)
    show(mask_red,"mask_red")
    return mask_red

def yellow_mask(input_image):
    hsv = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)
    yellow_lower = np.array([20, 170, 20])
    yellow_upper = np.array([40, 255, 255])  
    mask_yellow = cv2.inRange(hsv, yellow_lower, yellow_upper)
    show(mask_yellow,"mask_yellow")
    return mask_yellow

def black_mask(input_image):
    grey = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    # black_lower = np.array([0, 0, 0])
    # black_upper = np.array([20, 255, 255])  
    # mask_red = cv2.inRange(hsv, black_lower, black_upper)
    show(grey,"grey")
    ret, black_mask = cv2.threshold(grey,30,255,0)
    return black_mask

def extract_horizontal_belt(loaded_image, belt_height):
    """
    Extracts a horizontal belt of the specified height from the center of the given image.

    Parameters:
    - loaded_image: A NumPy array representing the loaded image.
    - belt_height: The desired height of the horizontal belt in pixels.

    Returns:
    - belt: A NumPy array representing the extracted horizontal belt.
    """
    
    # Get image dimensions
    height, width = loaded_image.shape[:2]
    
    # Calculate the starting and ending y-coordinates for the belt
    start_y = (height - belt_height) // 2
    end_y = start_y + belt_height
    
    # Extract the belt
    belt = loaded_image[start_y:end_y, :]
    
    return belt

def extract_below_horizontal_belt(loaded_image, belt_height):
    """
    Extracts a horizontal belt of the specified height from the center of the given image.

    Parameters:
    - loaded_image: A NumPy array representing the loaded image.
    - belt_height: The desired height of the horizontal belt in pixels.

    Returns:
    - belt: A NumPy array representing the extracted horizontal belt.
    """
    
    # Get image dimensions
    height, width = loaded_image.shape[:2]
    
    # Calculate the starting and ending y-coordinates for the belt
    start_y = (height - belt_height) // 2
    end_y = start_y + belt_height
    
    # Extract the belt
    below_belt = loaded_image[end_y:, :]
    
    return below_belt
def check_if_any_letter(image):
    global TURQOISE_COUNT
    global RED_COUNT
    global YELLOW_COUNT
    red_count = count_white_pixels(red_mask(image))
    turqoise_count = count_white_pixels(turqoise_mask(image))
    yellow_count = count_white_pixels(yellow_mask(image))
    if (red_count and turqoise_count) or (turqoise_count and yellow_count) or (yellow_count and red_count):
        print("Fatal error!")
    if red_count:
        print("Red!")
        RED_COUNT = RED_COUNT + 1
        # check_letter_b(red_mask(image))
        # check_letter_c(red_mask(image))
        result = ["Red","B","C"]
        if check_letter_b(red_mask(image)):
            print("B!")
        else:
            result[1] = "Not B"
        if check_letter_c(red_mask(image)):
            print("C!")
        else:
            result[2] = "Not C"
        results_list.append(result)
        return True
    if turqoise_count:
        print("Turqoise")
        TURQOISE_COUNT = TURQOISE_COUNT + 1
        # check_letter_b(turqoise_mask(image))
        # check_letter_c(turqoise_mask(image))
        result = ["Turqoise","B","C"]
        if check_letter_b(turqoise_mask(image)):
            print("B!")
        else:
            result[1] = "Not B"
        if check_letter_c(turqoise_mask(image)):
            print("C!")
        else:
            result[2] = "Not C"
        results_list.append(result)
        return True
    if yellow_count:
        print("Yellow")
        
        YELLOW_COUNT = YELLOW_COUNT + 1
        result = ["Yellow","B","C"]
        if check_letter_b(yellow_mask(image)):
            print("B!")
        else:
            result[1] = "Not B"
        if check_letter_c(yellow_mask(image)):
            print("C!")
        else:
            result[2] = "Not C"
        results_list.append(result)
        return True
    return False

def count_white_pixels(image):
    """
    Counts the white pixels in a grayscale image.

    Parameters:
    - image: A NumPy array representing the loaded grayscale image.

    Returns:
    - count: The number of white pixels in the image.
    """
    # Count pixels where the pixel value is 255 (white)
    count = np.sum(image == 255)
    return count
def count_black_pixels(image):
    """
    Counts the white pixels in a grayscale image.

    Parameters:
    - image: A NumPy array representing the loaded grayscale image.

    Returns:
    - count: The number of white pixels in the image.
    """
    # Count pixels where the pixel value is 255 (white)
    count = np.sum(image == 0)
    return count


def curtain_state(image):
    mask = black_mask(image)
    belt = extract_horizontal_belt(image,12)
    print("belt size px:")
    print(belt.shape[0]*belt.shape[1])

    mask_belt = black_mask(belt)
    mask_belt_size = mask_belt.shape[0]*mask_belt.shape[1]
    print(mask_belt.shape)
    # ret, mask_belt = cv2.threshold(mask_belt,127,255,0)
    white_pixels_in_belt = count_white_pixels(mask_belt)
    black_pixels_in_belt = count_black_pixels(mask_belt)
    print("white_pixels_in_belt")
    print(white_pixels_in_belt)
    show(mask,"2. mask")
    show(belt,"3. belt")
    show(mask_belt,"4. mask_belt")
    if white_pixels_in_belt > 2:
    # if check_if_any_letter(belt):
        print("Wiązka przecieta")
        return True
    else:
        print("Wiązka nie przecieta")
        return False
    
def check_letter_b(image):
    obraz_we = cv2.imread('PW_SW_9_ref.png') 
    # show(obraz_we,"1. bazowy obraz")
    b_mask = red_mask(obraz_we[:80, :70])
    # if count_white_pixels(image)<1250:
        # print("Not B")
        # print(count_white_pixels(image))
        # return
    scale_percent = 100*count_white_pixels(b_mask)/count_white_pixels(image) # percent of original size
    print("scale percentage: "+str(scale_percent))
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
  
    # resize image
    resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    ret, thresh = cv2.threshold(b_mask, 127, 255,0)
    ret, thresh2 = cv2.threshold(resized, 127, 255,0)
    contours,hierarchy = cv2.findContours(thresh,2,1)
    cnt1 = contours[0]
    contours,hierarchy = cv2.findContours(thresh2,2,1)
    cnt2 = contours[0]
    ret = cv2.matchShapes(cnt1,cnt2,1,0.0)
    print("b letter:")
    print(ret)
    print("wzor B shape:")
    print(b_mask.shape)
    show(b_mask,"wzor B")
    print("porownaj B shape:")
    print(image.shape)
    show(image,"porownaj B")
    show(resized,"resized porownaj B")
    if ret<0.05:
        print("To jest B!")
        return True
    else:
        return False


def check_letter_c(image):
    obraz_we = cv2.imread('PW_SW_9_ref.png') 
    c_mask = red_mask(obraz_we[:80, 240:340])
    # if count_white_pixels(image)>1250:
        # print("Not C")
        # print(count_white_pixels(image))
        # return
    scale_percent = 100*count_white_pixels(c_mask)/count_white_pixels(image) # percent of original size
    print("scale percentage: "+str(scale_percent))
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
  
    # resize image
    resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

    ret, thresh = cv2.threshold(c_mask, 127, 255,0)
    ret, thresh2 = cv2.threshold(resized, 127, 255,0)
    contours,hierarchy = cv2.findContours(thresh,2,1)
    cnt1 = contours[0]
    contours,hierarchy = cv2.findContours(thresh2,2,1)
    cnt2 = contours[0]
    ret = cv2.matchShapes(cnt1,cnt2,1,0.0)
    print("c letter:")
    print(ret)
    show(c_mask,"wzor C")
    print("wzor C shape:")
    print(c_mask.shape)
    print("porownaj C shape:")
    print(image.shape)
    show(image,"porownaj C")
    show(resized,"resized porownaj C")
    if ret<0.05:
        return True
    else:
        return False

    

# obraz_we = cv2.imread('PW_SW_9_ref.png') 
# show(obraz_we,"1. bazowy obraz")
# b_mask = red_mask(obraz_we[:80, :70])#1619
# b_mask = turqoise_mask(obraz_we[80:160, :70])#1391
# b_mask = yellow_mask(obraz_we[160:240, :70])#1619

#złe:
# b_mask = red_mask(obraz_we[:80, 70:160])#1581

# b_mask = turqoise_mask(obraz_we[80:160, 70:160])#1317
# b_mask = yellow_mask(obraz_we[160:240, 70:160])#1581

#dobre
# c_mask = red_mask(obraz_we[:80, 240:340])#1106
# c_mask2 = turqoise_mask(obraz_we[80:160, 240:340])#937
# c_mask2 = yellow_mask(obraz_we[160:240,  240:340])#1106

#złe:
# c_mask_bad = red_mask(obraz_we[:80, 340:440])#1590

# c_mask = turqoise_mask(obraz_we[80:160, 340:440])#1317
# c_mask = yellow_mask(obraz_we[160:240, 340:440])#1581

# ret, thresh = cv2.threshold(c_mask, 127, 255,0)
# ret, thresh2 = cv2.threshold(b_mask, 127, 255,0)
# contours,hierarchy = cv2.findContours(thresh,2,1)
# cnt1 = contours[0]
# contours,hierarchy = cv2.findContours(thresh2,2,1)
# cnt2 = contours[0]
# ret = cv2.matchShapes(cnt1,cnt2,1,0.0)
# print(ret)

# print(count_white_pixels(c_mask))

# show(thresh,"2. wzor")
# show(thresh2,"2. porownanie")
# show(c_mask_bad,"2. litera zła")
# contours2, hierarchy2 = cv2.findContours(b_mask,  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
# for countour in contours2:
#     area = cv2.contourArea(countour)
#     print(area)


# last_curtain_state =  True
# current_curtain_state = curtain_state(obraz_we)
# cv2.waitKey(0)
# exit()

wideo = cv2.VideoCapture('PW_SW_9.avi')


last_curtain_state =  False
trigger = False
while(wideo.isOpened()):
    read_ok, frame = wideo.read()
    if read_ok:  # udany odczyt ramki
        
        if trigger:
            print("Trigger - let's analyze")
            #check color
            to_analyze = extract_below_horizontal_belt(frame,12)
            check_if_any_letter(to_analyze)
            # cv2.imshow('to_analyze',to_analyze)
            show(to_analyze,'to_analyze')
            # print(results_list)
            print('\n'.join(map(str,results_list )))




        
        # cv2.imshow('frame',frame)
        show(frame,'frame')


        current_curtain_state = curtain_state(frame)
        trigger = False
        if last_curtain_state == True and current_curtain_state == False:
            print("Time to snapshot")
            trigger = True

        last_curtain_state = current_curtain_state

        cv2.waitKey(2)   # jedna klatka na 33ms = 30 fps
        # if cv2.waitKey(1) == ord(' '):
            # print("Emergency STOP! Press space to continue")
            # while cv2.waitKey(1000) != ord(' '):
                # print("Press space to continue")
        # cv2.waitKey(0)   # czekamy na wcisniecie klawisz po kazdej klatce
        # if keyboard.read_key() == 'space':
            # print("A Key Pressed") 
            # exit()
    else:   # koniec pliku
        wideo.release()            
        cv2.waitKey(1) 
        cv2.destroyAllWindows()
        cv2.waitKey(1)



print("TURQOISE:     " + str(TURQOISE_COUNT))
print("RED COUNT:    " + str(RED_COUNT))
print("YELLOW COUNT: " + str(YELLOW_COUNT))
# print(results_list)
print('\n'.join(map(str,results_list )))


#2. Wykonaj segmentację koloru obrazu wejściowego, w wyniku której powstanie obraz binarny zawierający wszystkie obszary mapy o kolorze takim jak kolor jakim zaznaczono województwo, które masz wyodrębnić z obrazu wejściowego (województwo referencyjne)-> 5 pkt

#117 175 104
# show(obraz_we,"1. bazowy obraz")

cv2.waitKey(0)

# C turkus ok
# C turkus ok
# B czerwone nie ok
# B czerwone ok
# B czerwone ok
# B żółte ok
# B żółte nie ok
# C czerwone nie ok
# C turkus ok
# B żółte ok
# B żółte ok
# B czerwone ok
# B czerwone ok
# C turkus ok
# B czerwone nie ok
# B żółte ok
# B żółte ok
# B turkusowe ok
# C żółte nie ok
# B turkusowe nie ok
# C turkus ok
# B żółte ok