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



def not_ok_learning_set(image):
    letter_height = 80
    letter_width = 90
    features = []
    labels = []
    augmented_letters = []
    for row in range(3):  # First and fourth columns are correct
        for col in [1,2,4,5]:
            letter_image = image[row * letter_height:(row + 1) * letter_height, col * letter_width:(col + 1) * letter_width]
            letter_image = crop(letter_image)
            # augmented_letters = augment_letter(letter_image, num_augmentations=500)
            augmented_letters.append(letter_image)
            # show(letter_image,"not_ok_learning_set"+str(row)+":"+str(col))
            # cv2.waitKey(0)
    return augmented_letters

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

def crop(_image):
    ret, thresh = cv2.threshold(_image, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh,  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
    # print(contours[0])
    x, y, w, h = cv2.boundingRect(contours[0])

    # Crop the image using array slicing
    cropped_image = _image[y:y+h, x:x+w]
    return cropped_image

def extract_features_and_labels(image):
    # Load the image in grayscale
    # image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Define the correct positions of letters in the training image
    # We assume each letter is of the same size and non-overlapping
    # You will need to adjust these values according to your actual image
    letter_height = 80
    letter_width = 90
    features = []
    labels = []
    for row in range(3):
        for col in range(3):  # First and fourth columns are correct
            # Extract letter image
            # letter_image = image[row * letter_height:(row + 1) * letter_height, col * letter_width:(col + 1) * letter_width]
            letter_image = image[row * letter_height:(row + 1) * letter_height, col * letter_width:(col + 1) * letter_width]
            augmented_letters = []
            augmented_letters.extend(augment_letter(letter_image, num_augmentations=50))
            for idx, aug_image in enumerate(augmented_letters):
                feature = preprocess_letter(aug_image)
                if col == 0 or col == 3:
                    if col == 0:
                        print("B")
                        features.append(feature)
                        labels.append("B")
                        # show(letter_image,"letter_image"+str(row)+":"+str(col)+":"+str(idx)+"B")
                    if col == 3:
                        features.append(feature)
                        labels.append("C")
                        # show(letter_image,"letter_image"+str(row)+":"+str(col)+":"+str(idx)+"C")
                        print("C")
                elif row < 3:
                    print("not ok B")
                    # features.append(feature)
                    # labels.append("not ok B")
                    # show(letter_image,"letter_image"+str(row)+":"+str(col)+":"+str(idx)+"not ok B")
                else: 
                    print("not ok C")
                    # features.append(feature)
                    # labels.append("not ok C")
                    # show(letter_image,"letter_image"+str(row)+":"+str(col)+":"+str(idx)+"not ok C")

            # Flatten the letter image to a 1D array
            # feature = letter_image.flatten()
            # feature = preprocess_letter(letter_image)
            
            # Label 'B' for the first row, and similarly for other rows
            # if col == 0 or col == 3:
            #     if col == 0:
            #         features.append(feature)
            #         labels.append("B")
            #         show(letter_image,"letter_image"+str(row)+":"+str(col)+"B")
            #     if col == 3:
            #         features.append(feature)
            #         labels.append("C")
            #         show(letter_image,"letter_image"+str(row)+":"+str(col)+"C")
            #         print("0")
            # else:
            #     print("0")
            #     # features.append(feature)
            #     # labels.append("not ok")
            #     show(letter_image,"letter_image"+str(row)+":"+str(col)+"not ok")

            # labels.append('B' if row == 0 else 'C' if row == 1 else 'not ok')
            
    return np.array(features), np.array(labels)

def train_classifier(features, labels):
    # knn = KNeighborsClassifier(n_neighbors=3)
    # knn = SVC(gamma='auto')
    knn = MLPClassifier(hidden_layer_sizes=(100, ), max_iter=1000)
    knn.fit(features, labels)

    # knn.fit(features, labels)
    # knn.fit(features, labels)
    return knn

def classify_new_image(classifier, new_image):
    # Load and preprocess the new image similarly to the training images
    # new_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Assuming the new image is a single letter of the same size as the training letters
    # feature = new_image.flatten().reshape(1, -1)
    feature = preprocess_letter(new_image).reshape(1, -1)
    show(new_image,"clasify")
    # Predict the letter
    prediction = classifier.predict(feature)
    return prediction

def load_datasets(images_b, images_c, images_not_ok):
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
    for img in images_not_ok:
        if img is not None:
            img = cv2.resize(img, (64, 64)) # Resize for uniformity
            images.append(img)
            labels.append("not ok")
    return np.array(images), np.array(labels)

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

def preprocess_image(new_image, target_size=(64, 64)):
    # Resize the image to match the input shape expected by the model
    img_resized = cv2.resize(new_image, target_size)  # Assuming using OpenCV for resizing
    
    # Flatten the image if the model expects 2D input
    img_flattened = img_resized.reshape(1, -1)  # 1 for a single image, -1 to flatten
    
    # Normalize pixel values if your model expects normalized inputs
    # img_normalized = img_flattened.astype('float32') / 255.0
    
    return img_flattened

def crop(_image):
    ret, thresh = cv2.threshold(_image, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh,  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
    # print(contours[0])
    x, y, w, h = cv2.boundingRect(contours[0])

    # Crop the image using array slicing
    cropped_image = _image[y:y+h, x:x+w]
    return cropped_image
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
        return
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
    # return False

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

wideo = cv2.VideoCapture('PW_SW_9.avi')

# gnb_loaded = load('gnb_model.joblib')
# nc_loaded = load('nc_model.joblib')
# lr_load = load('lr_model.joblib')
learn_img = cv2.imread('PW_SW_9_ref.png') 
show(learn_img,"1. learn_img")
black_mask_learn_img = black_mask(learn_img)
_images_c = letter_C_learning_set(black_mask_learn_img)
_images_b = letter_B_learning_set(black_mask_learn_img)
_not_ok_images = not_ok_learning_set(black_mask_learn_img)
images, labels = load_datasets(_images_b, _images_c,_not_ok_images)


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
while(wideo.isOpened()):
    read_ok, frame = wideo.read()
    if read_ok:  # udany odczyt ramki
        
        if trigger:
            print("Trigger - let's analyze")
            #check color
            to_analyze = extract_below_horizontal_belt(frame,12)
            results_list.append(check_if_any_letter(to_analyze))
            # cv2.imshow('to_analyze',to_analyze)
            show(to_analyze,'to_analyze' + str(to_analyze_counter))
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
        cv2.waitKey(1) 
        cv2.destroyAllWindows()
        cv2.waitKey(1)



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