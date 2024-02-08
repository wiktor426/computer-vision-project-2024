import cv2
import numpy as np

TURQOISE_COUNT = 0
RED_COUNT = 0
YELLOW_COUNT = 0

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
        return True
    if turqoise_count:
        print("Turqoise")
        TURQOISE_COUNT = TURQOISE_COUNT + 1
        return True
    if yellow_count:
        print("Yellow")
        YELLOW_COUNT = YELLOW_COUNT + 1
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
    

    


# obraz_we = cv2.imread('PW_SW_9_ref.png') 

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
            cv2.imshow('to_analyze',to_analyze)


        
        cv2.imshow('frame',frame)


        current_curtain_state = curtain_state(frame)
        trigger = False
        if last_curtain_state == True and current_curtain_state == False:
            print("Time to snapshot")
            trigger = True

        last_curtain_state = current_curtain_state

        # cv2.waitKey(10)   # jedna klatka na 33ms = 30 fps
        cv2.waitKey(0)   # czekamy na wcisniecie klawisz po kazdej klatce
    else:   # koniec pliku
        wideo.release()            
        cv2.waitKey(1) 
        cv2.destroyAllWindows()
        cv2.waitKey(1)

print("TURQOISE:     " + str(TURQOISE_COUNT))
print("RED COUNT:    " + str(RED_COUNT))
print("YELLOW COUNT: " + str(YELLOW_COUNT))

#2. Wykonaj segmentację koloru obrazu wejściowego, w wyniku której powstanie obraz binarny zawierający wszystkie obszary mapy o kolorze takim jak kolor jakim zaznaczono województwo, które masz wyodrębnić z obrazu wejściowego (województwo referencyjne)-> 5 pkt

#117 175 104
# show(obraz_we,"1. bazowy obraz")

cv2.waitKey(0)