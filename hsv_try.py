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

def red_mask(input_image):
    hsv = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)
    red_lower = np.array([160, 170, 20])
    red_upper = np.array([180, 255, 255])  
    mask_red = cv2.inRange(hsv, red_lower, red_upper)
    return mask_red

def yellow_mask(input_image):
    hsv = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)
    yellow_lower = np.array([20, 170, 20])
    yellow_upper = np.array([40, 255, 255])  
    mask_red = cv2.inRange(hsv, yellow_lower, yellow_upper)
    return mask_red

def black_mask(input_image):
    hsv = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)
    black_lower = np.array([0, 0, 0])
    black_upper = np.array([20, 255, 255])  
    mask_red = cv2.inRange(hsv, black_lower, black_upper)
    return mask_red

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


def curtain_state(image):
    mask = black_mask(obraz_we)
    belt = extract_horizontal_belt(image,20)
    print("belt size px:")
    print(belt.shape[0]*belt.shape[1])

    mask_belt = black_mask(belt)
    mask_belt_size = mask_belt.shape[0]*mask_belt.shape[1]
    print(mask_belt.shape)
    # ret, mask_belt = cv2.threshold(mask_belt,127,255,0)
    white_pixels_in_belt = count_white_pixels(mask_belt)
    print("white_pixels_in_belt")
    print(white_pixels_in_belt)
    show(mask,"2. mask")
    show(belt,"3. belt")
    show(mask_belt,"4. mask_belt")
    if white_pixels_in_belt > mask_belt_size*0.95:
        print("Wiązka nie przecieta")
        return False
    else:
        print("Wiązka przecieta")
        return True


obraz_we = cv2.imread('PW_SW_9_ref.png') 

last_curtain_state =  True
current_curtain_state = curtain_state(obraz_we)
if current_curtain_state:
    print("Produkt")
else:
    print("Przerwa")
#2. Wykonaj segmentację koloru obrazu wejściowego, w wyniku której powstanie obraz binarny zawierający wszystkie obszary mapy o kolorze takim jak kolor jakim zaznaczono województwo, które masz wyodrębnić z obrazu wejściowego (województwo referencyjne)-> 5 pkt

#117 175 104
show(obraz_we,"1. bazowy obraz")

cv2.waitKey(0)