from keras.models import load_model 
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import img_to_array, array_to_img
from PIL import Image
from imagehash import phash
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.losses import mean_squared_error, categorical_crossentropy
import matplotlib.pyplot as plt
import cv2
from keras.utils import to_categorical
import scipy.misc

model = load_model("model.h5")

sess = K.get_session()

K.set_session(sess)

TREE_FROG_IDX = 31

IMAGE_DIMS = (224, 224,3)

eps = 0.02

#model.summary()

def prepare_image(image, target=IMAGE_DIMS):
    # if the image mode is not RGB, convert it
    if image.mode != "RGB":
        image = image.convert("RGB")

    # resize the input image and preprocess it
    image = img_to_array(image)
    image = cv2.resize(image, (224,224)) #Differs from image.resize used
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    # return the processed image
    return image

image = Image.open("img/trixi.png")

#plt.imshow(image1)

image_p = prepare_image(image)

#plt.imshow(image_p)

#print(image_p1)

#plt.imshow(image_p1)

#Max 225

imaget = Image.open("new_img.png")

imaget = prepare_image(imaget)

print(np.argmax(model.predict(imaget)))

print(model.predict(imaget)[0][31])

input_tensor = model.input
output_tensor = model.output

def hash_hamming_distance(h1, h2):
    s1 = str(h1)
    s2 = str(h2)
    return sum(map(lambda x: 0 if x[0] == x[1] else 1, zip(s1, s2)))


def is_similar_img(path1, path2):
    image1 = Image.open(path1)
    image2 = Image.open(path2)

    dist = hash_hamming_distance(phash(image1), phash(image2))
    return dist <= 2, dist

print("Here")

sim, dist = is_similar_img("new_img.png", "img/trixi.png")

print(dist)

if sim:
    
    print("Works")
    
else:
    
    print("Doesn't")

def adv_noise_step(x, pred, false_label = TREE_FROG_IDX):

    false_label = K.one_hot(false_label, 1000)
    false_label = K.expand_dims(false_label, axis = 0)
    
    loss = mean_squared_error(false_label, pred) #Consider adding custom loss to account for change from base image (Add hash hemming distance and multiply by value x)

    grad = K.gradients(loss, x)[0]

    noise = eps*K.sign(K.gradients(loss, x)[0])
    
    adv_x = x - noise
    
    adv_x = K.clip(adv_x, -1.0, 1.0)
    
    print("occur")
    
    return K.stop_gradient(adv_x), noise, grad

adv_noise = np.load("noise.npy")

adv_image = image_p

adv_image -= adv_noise

#save = cv2.resize(adv_image, (1080, 1920))

#scipy.misc.imsave("new_img.png", np.squeeze(save))

'''
for i in range(100):
    
    a,b,c = adv_noise_step(input_tensor, output_tensor)  
    adv_image, noise, grad = sess.run((a,b,c), {input_tensor: adv_image})
    adv_noise += noise
    #print(noise)
    if (np.argmax(model.predict(adv_image)) == 31):
        np.save('noise.npy', adv_noise)
        scipy.misc.imsave('altered_img.png', np.squeeze(adv_image))
        scipy.misc.imsave('noise.png', np.squeeze(adv_noise))
        break
        
    print("Category: ", np.argmax(model.predict(adv_image)))
    print("Percentile: ",np.max(model.predict(adv_image)))

image_p1 = cv2.resize(image_p, (1080, 1920))
'''
