import cv2
import time
import random

while True:
    original_image = cv2.imread('denoising.png')
    random_size = [random.randint(256, 1024), random.randint(256, 1024)]
    t0 = time.time()
    resized_image = cv2.resize(original_image, random_size)
    t1 = time.time()
    print('resize time: %.3f ms' % ((t1 - t0) * 1000))