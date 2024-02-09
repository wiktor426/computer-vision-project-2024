import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import NearestCentroid
from joblib import dump, load


def show(image,title=""):
    # if obraz.ndim == 2:
        # cv2.imshow(obraz,cmap='gray')
    # else:
    cv2.namedWindow(title, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.imwrite(title + ".jpg", image) 
    cv2.imshow(title,image)


def preprocess_image(new_image, target_size=(64, 64)):
    # Resize the image to match the input shape expected by the model
    img_resized = cv2.resize(new_image, target_size)  # Assuming using OpenCV for resizing
    
    # Flatten the image if the model expects 2D input
    img_flattened = img_resized.reshape(1, -1)  # 1 for a single image, -1 to flatten
    
    # Normalize pixel values if your model expects normalized inputs
    # img_normalized = img_flattened.astype('float32') / 255.0
    
    return img_flattened


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

def letter_B_learning_set(image):
    letter_height = 80
    letter_width = 90
    features = []
    labels = []
    augmented_letters = []
    for row in range(3):
        col = 0
        letter_image = image[row * letter_height:(row + 1) * letter_height, col * letter_width:(col + 1) * letter_width]
        augmented_letters = augment_letter(letter_image, num_augmentations=50)
    return augmented_letters

def letter_test_set(image):
    augmented_letters = []
    letter_image = image
    augmented_letters = augment_letter(letter_image, num_augmentations=50)
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
            augmented_letters = augment_letter(letter_image, num_augmentations=50)
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
        augmented_letters = augment_letter(letter_image, num_augmentations=50)
    return augmented_letters

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

def load_test_sets(images_b, images_c):
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
    return np.array(images), np.array(labels)
def black_mask(input_image):
    grey = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    # black_lower = np.array([0, 0, 0])
    # black_upper = np.array([20, 255, 255])  
    # mask_red = cv2.inRange(hsv, black_lower, black_upper)
    show(grey,"grey")
    ret, black_mask = cv2.threshold(grey,30,255,0)
    return black_mask

# Load dataset
folder = 'dataset'
learn_img = cv2.imread('PW_SW_9_ref.png') 
show(learn_img,"1. learn_img")
black_mask_learn_img = black_mask(learn_img)


test_b_img = cv2.imread('clasifyB.jpg') 
black_mask_test_B = black_mask(test_b_img)
b_test_set = letter_test_set(black_mask_test_B)

test_c_img = cv2.imread('clasifyC.jpg') 
black_mask_test_C = black_mask(test_c_img)
c_test_set = letter_test_set(black_mask_test_C)


_images_c = letter_C_learning_set(black_mask_learn_img)
_images_b = letter_B_learning_set(black_mask_learn_img)
_not_ok_images = not_ok_learning_set(black_mask_learn_img)
images, labels = load_datasets(_images_b, _images_c,_not_ok_images)
test_images, test_labels = load_test_sets(b_test_set, c_test_set)

num_samples, height, width = images.shape
X = images.reshape(num_samples, -1)  # This flattens each image

num_samples, height, width = test_images.shape
X_test = test_images.reshape(num_samples, -1)
y_test = test_labels

# Assuming 'labels' is your array of labels
y = labels

# X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = X
y_train = y

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
knn_score = knn.score(X_test, y_test)
print(f'KNeighborsClassifier accuracy: {knn_score}')

gnb = GaussianNB()
gnb.fit(X_train, y_train)
gnb_score = gnb.score(X_test, y_test)
print(f'GaussianNB accuracy: {gnb_score}')




nc = NearestCentroid()
nc.fit(X_train, y_train)
nc_score = nc.score(X_test, y_test)
print(f'NearestCentroid accuracy: {nc_score}')


test_solo_B = preprocess_image(black_mask_test_B)
test_solo_C = preprocess_image(black_mask_test_C)
predicted_label = gnb.predict(test_solo_B)
print(f"Predicted label: {predicted_label}")

predicted_label = gnb.predict(test_solo_C)
print(f"Predicted label: {predicted_label}")

save_model = input()
if(save_model == 'yes'):
    dump(gnb, 'gnb_model.joblib')