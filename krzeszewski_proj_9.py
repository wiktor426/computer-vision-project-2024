import cv2
import numpy as np
import keyboard
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from joblib import dump, load



def augment_letter(image, num_augmentations=5):
    """
    Generate augmented images from a single letter image.

    Parameters:
    - image: A single letter image.
    - num_augmentations: Number of augmented images to generate.

    Returns:
    - aug_images: List of augmented images.
    """
    aug_images = []
    rows, cols = image.shape

    for _ in range(num_augmentations):
        # Randomly choose the type of transformation
        transformation_type = np.random.choice(['rotate', 'translate', 'scale', 'flip', 'noise'])

        if transformation_type == 'rotate':
            # Rotate the image by a random angle between -15 and 15 degrees
            angle = np.random.uniform(-15, 15)
            M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
            dst = cv2.warpAffine(image, M, (cols, rows))
        
        elif transformation_type == 'translate':
            # Translate the image by a random offset
            tx = np.random.uniform(-5, 5)
            ty = np.random.uniform(-5, 5)
            M = np.float32([[1, 0, tx], [0, 1, ty]])
            dst = cv2.warpAffine(image, M, (cols, rows))
        
        elif transformation_type == 'scale':
            # Scale the image by a random factor between 0.9 and 1.1
            fx = np.random.uniform(0.9, 1.1)
            fy = np.random.uniform(0.9, 1.1)
            dst = cv2.resize(image, None, fx=fx, fy=fy, interpolation=cv2.INTER_LINEAR)
        
        elif transformation_type == 'flip':
            # Flip the image horizontally or vertically
            flip_type = np.random.choice([-1, 0, 1])  # -1: flip both, 0: vertical, 1: horizontal
            dst = cv2.flip(image, flip_type)
        
        elif transformation_type == 'noise':
            # Add random noise to the image
            gauss = np.random.normal(0, 1, image.size)
            gauss = gauss.reshape(image.shape).astype('uint8')
            dst = cv2.add(image, gauss)
        
        # Add the transformed image to the list
        aug_images.append(dst)

    return aug_images

# Function to preprocess the image, resize and flatten the letter
def preprocess_letter(image, size=(50, 50)):
    # Resize the image to ensure consistency
    resized_image = cv2.resize(image, size)
    # Flatten the image to create a feature vector
    flat_features = resized_image.flatten()
    return flat_features

# Learning set preparation function for B letter
def letter_B_learning_set(image):
    letter_height = 80
    letter_width = 90
    features = []
    labels = []
    augmented_letters = []
    for row in range(3):
        col = 0
        letter_image = image[row * letter_height:(row + 1) * letter_height, col * letter_width:(col + 1) * letter_width]
        letter_image = crop(letter_image)
        # augmented_letters = augment_letter(letter_image, num_augmentations=50)
        augmented_letters.append(letter_image)
    return augmented_letters


# Learning set preparation function for not B letter
def not_ok_B_learning_set(image):
    letter_height = 80
    letter_width = 90
    features = []
    labels = []
    augmented_letters = []
    for row in range(3):  # First and fourth columns are correct
        for col in [1,2]:
            letter_image = image[row * letter_height:(row + 1) * letter_height, col * letter_width:(col + 1) * letter_width]
            letter_image = crop(letter_image)
            # augmented_letters = augment_letter(letter_image, num_augmentations=500)
            augmented_letters.append(letter_image)
            # show(letter_image,"not_ok_learning_set"+str(row)+":"+str(col))
            # cv2.waitKey(0)
    return augmented_letters

# Learning set preparation function for not C letter
def not_ok_C_learning_set(image):
    letter_height = 80
    letter_width = 90
    features = []
    labels = []
    augmented_letters = []
    for row in range(3):  # First and fourth columns are correct
        for col in [4,5]:
            letter_image = image[row * letter_height:(row + 1) * letter_height, col * letter_width:(col + 1) * letter_width]
            letter_image = crop(letter_image)
            # augmented_letters = augment_letter(letter_image, num_augmentations=500)
            augmented_letters.append(letter_image)
            # show(letter_image,"not_ok_learning_set"+str(row)+":"+str(col))
            # cv2.waitKey(0)
    return augmented_letters

# Learning set preparation function for not C letter
def letter_C_learning_set(image):
    letter_height = 80
    letter_width = 90
    features = []
    labels = []
    augmented_letters = []
    for row in range(3):  # First and fourth columns are correct
        col = 3
        letter_image = image[row * letter_height:(row + 1) * letter_height, col * letter_width:(col + 1) * letter_width]
        letter_image = crop(letter_image)
        # augmented_letters = augment_letter(letter_image, num_augmentations=50)
        augmented_letters.append(letter_image)
    return augmented_letters

# Load datasets and set labels
def load_datasets(images_b, images_c, images_not_ok_b, images_not_ok_c):
    images = []
    labels = []
    for img in images_b:
        if img is not None:
            img = cv2.resize(img, (64, 64)) # Resize for uniformity
            images.append(img)
            labels.append("B")
    for img in images_c:
        if img is not None:
            img = cv2.resize(img, (64, 64)) # Resize for uniformity
            images.append(img)
            labels.append("C")
    for img in images_not_ok_b:
        if img is not None:
            img = cv2.resize(img, (64, 64)) # Resize for uniformity
            images.append(img)
            labels.append("not ok b")
    for img in images_not_ok_c:
        if img is not None:
            img = cv2.resize(img, (64, 64)) # Resize for uniformity
            images.append(img)
            labels.append("not ok c")
    return np.array(images), np.array(labels)


# Shows image with title
def show(image,title=""):
    # if obraz.ndim == 2:
        # cv2.imshow(obraz,cmap='gray')
    # else:
    cv2.namedWindow(title, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.imwrite(title + ".jpg", image) 
    cv2.imshow(title,image)
    # plt.title(tytul)   

# Masks everything turqoise to white
def turqoise_mask(input_image):
    hsv = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)
    turqoise_lower = np.array([80, 200, 160])
    turqoise_upper = np.array([110, 255, 255])  
    mask_turqoise = cv2.inRange(hsv, turqoise_lower, turqoise_upper)
    show(mask_turqoise,"mask_turqoise")
    return mask_turqoise

# Masks everything red to white
def red_mask(input_image):
    hsv = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)
    red_lower = np.array([160, 170, 20])
    red_upper = np.array([180, 255, 255])  
    mask_red = cv2.inRange(hsv, red_lower, red_upper)
    show(mask_red,"mask_red")
    return mask_red
# Masks everything yellow to white
def yellow_mask(input_image):
    hsv = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)
    yellow_lower = np.array([20, 170, 20])
    yellow_upper = np.array([40, 255, 255])  
    mask_yellow = cv2.inRange(hsv, yellow_lower, yellow_upper)
    show(mask_yellow,"mask_yellow")
    return mask_yellow

# Masks everything not black to white
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

# Extracts image under the horizontal belt "curtain"
def extract_below_horizontal_belt(loaded_image, belt_height):    
    # Get image dimensions
    height, width = loaded_image.shape[:2]
    
    # Calculate the starting and ending y-coordinates for the belt
    start_y = (height - belt_height) // 2
    end_y = start_y + belt_height
    
    # Extract the belt
    below_belt = loaded_image[end_y:, :]
    
    return below_belt

# Function to resize and flatten image
def preprocess_image(new_image, target_size=(64, 64)):
    # Resize the image to match the input shape expected by the model
    img_resized = cv2.resize(new_image, target_size)  # Assuming using OpenCV for resizing
    
    # Flatten the image if the model expects 2D input
    img_flattened = img_resized.reshape(1, -1)  # 1 for a single image, -1 to flatten
    
    # Normalize pixel values if your model expects normalized inputs
    # img_normalized = img_flattened.astype('float32') / 255.0
    
    return img_flattened

# Function to crop the letter
def crop(_image):
    ret, thresh = cv2.threshold(_image, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh,  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
    # print(contours[0])
    x, y, w, h = cv2.boundingRect(contours[0])

    # Crop the image using array slicing
    cropped_image = _image[y:y+h, x:x+w]
    return cropped_image
# Functions check if this is letter, checks color and classify.
# Return result as array: 
# 1. Colour
# 2. Classification
def check_if_any_letter(image):
    global TURQOISE_COUNT
    global RED_COUNT
    global YELLOW_COUNT
    global predicted_label
    global lr_load
    red_count = count_white_pixels(red_mask(image))
    turqoise_count = count_white_pixels(turqoise_mask(image))
    yellow_count = count_white_pixels(yellow_mask(image))

    black_mask_image = black_mask(image)
    black_mask_image = crop(black_mask_image)
    black_mask_image = cv2.resize(black_mask_image, (64, 64))
    predicted_label = lr_load.predict(preprocess_image(black_mask_image))
    if (red_count and turqoise_count) or (turqoise_count and yellow_count) or (yellow_count and red_count):
        print("Fatal error!")
        return False
    if red_count:
        print("Red!")
        RED_COUNT = RED_COUNT + 1

        
        result = ["Red", f"{predicted_label}"]
        return result
    if turqoise_count:
        print("Turqoise")
        TURQOISE_COUNT = TURQOISE_COUNT + 1
        result = ["Turqoise", f"{predicted_label}"]
        return result

    if yellow_count:
        print("Yellow")
        
        YELLOW_COUNT = YELLOW_COUNT + 1
        result = ["Yellow", f"{predicted_label}"]
        return result
    return False

# Function to add two centered texts on image
def write_centered_text(image, text1, text2, font_scale, font=cv2.FONT_HERSHEY_SIMPLEX, color=(255, 255, 255), thickness=2):
    # Copy the input image to not overwrite the original one
    img = image.copy()
    
    # Get the width and height of the image
    img_height, img_width = img.shape[:2]
    
    # Calculate the total height of the text block (two lines of text)
    ((text_width1, text_height1), _) = cv2.getTextSize(text1, font, font_scale, thickness)
    ((text_width2, text_height2), _) = cv2.getTextSize(text2, font, font_scale, thickness)
    total_text_height = text_height1 + text_height2
    
    # Calculate the starting Y position such that the text block will be centered
    y_position = (img_height + total_text_height) // 2 - text_height2
    
    # Calculate the X positions such that the texts will be centered
    x_position1 = (img_width - text_width1) // 2
    x_position2 = (img_width - text_width2) // 2
    
    # Put the first line of text on the image
    img = cv2.putText(img, text1, (x_position1, y_position - 10), font, font_scale, color, thickness)
    
    # Put the second line of text on the image
    img = cv2.putText(img, text2, (x_position2, y_position + text_height1 + 10), font, font_scale, color, thickness)
    
    return img

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

# Function checking if curtain is cut or not
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
        print("Curtain cut")
        return True
    else:
        print("Curtain not cut")
        return False
    



TURQOISE_COUNT = 0
RED_COUNT = 0
YELLOW_COUNT = 0
results_list = []

wideo = cv2.VideoCapture('PW_SW_9.avi')

# gnb_loaded = load('gnb_model.joblib')
# nc_loaded = load('nc_model.joblib')
# lr_load = load('lr_model.joblib')
learn_img = cv2.imread('PW_SW_9_ref.png') 
show(learn_img,"1. learn_img")
black_mask_learn_img = black_mask(learn_img)
_images_c = letter_C_learning_set(black_mask_learn_img)
_images_b = letter_B_learning_set(black_mask_learn_img)
_not_ok_b_images = not_ok_B_learning_set(black_mask_learn_img)
_not_ok_c_images = not_ok_C_learning_set(black_mask_learn_img)
images, labels = load_datasets(_images_b, _images_c,_not_ok_b_images,_not_ok_c_images)


num_samples, height, width = images.shape
X = images.reshape(num_samples, -1)  # This flattens each image



y = labels

X_train = X
y_train = y
lr_load = LogisticRegression(max_iter=5000)  # Increase max_iter if needed for convergence
lr_load.fit(X_train, y_train)


last_curtain_state =  False
trigger = False
to_analyze_counter = 0
font_scale = 0.5
while(wideo.isOpened()):
    read_ok, frame = wideo.read()
    if read_ok:  # udany odczyt ramki
        
        if trigger:
            print("Trigger - let's analyze")
            #check color
            to_analyze = extract_below_horizontal_belt(frame,12)
            result = check_if_any_letter(to_analyze)
            if result:
                results_list.append(result)
            # cv2.imshow('to_analyze',to_analyze)
                output_image = write_centered_text(to_analyze, result[0], result[1], font_scale)
                show(output_image,'Produkt nr' + str(to_analyze_counter))
                to_analyze_counter = to_analyze_counter + 1
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

        # cv2.waitKey(2)   # jedna klatka na 33ms = 30 fps
        if cv2.waitKey(1) == ord(' '):
            print("Emergency STOP! Press space to continue")
            while cv2.waitKey(1000) != ord(' '):
                print("Press space to continue")
        # cv2.waitKey(0)   # czekamy na wcisniecie klawisz po kazdej klatce
        # if keyboard.read_key() == 'space':
            # print("A Key Pressed") 
            # exit()
    else:   # koniec pliku
        wideo.release()            
        cv2.waitKey(0) 
        cv2.destroyAllWindows()
        cv2.waitKey(1)


print("")
print("SUMMARY:")
print("TURQOISE:     " + str(TURQOISE_COUNT))
print("RED COUNT:    " + str(RED_COUNT))
print("YELLOW COUNT: " + str(YELLOW_COUNT))
# print(results_list)
print('\n'.join(map(str,results_list )))


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