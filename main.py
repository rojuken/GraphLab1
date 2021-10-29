import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def find_ill_parts(img):
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))
    image_erode = cv.erode(img, kernel)

    hsv_img = cv.cvtColor(image_erode, cv.COLOR_BGR2HSV)
    healthy_part = cv.inRange(hsv_img, (35,24,36), (86, 255, 255))

    markers = np.zeros((img.shape[0], img.shape[1]), dtype="int32")
    markers[healthy_part>1] = 255
    markers[236:255, 0:20] = 1
    markers[0:20, 0:20] = 1
    markers[0:20, 236:255] = 1
    markers[236:255, 236:255] = 1
    leafs_area_BGR = cv.watershed(image_erode, markers)
    ill_part = leafs_area_BGR - healthy_part
    mask = np.zeros_like(img, np.uint8)
    mask [leafs_area_BGR > 1] = (255 , 0, 255)
    mask[ill_part > 1] = (0, 0, 255)
    return mask


def change_shadows(img):
    hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    b, g, r = cv.split(img[0:20, 235:255])
    b2, g2, r2 = cv.split(img[235:255, 0:20])
    b = np.append(b, b2, 1)
    g = np.append(g, g2, 1)
    r = np.append(r, r2, 1)
    black_pixels = np.where(
        (hsv_img[:, :, 2] < 50)
    )
    img[black_pixels] = [np.median(b), np.median(g), np.median(r)]
    return img

rows = 1
columns = 3
fig =  plt.figure(figsize=(10, 7))
img = cv.imread("2.jpg", cv.IMREAD_COLOR)
wihout_shadows = change_shadows(img)

gauss = cv.GaussianBlur(wihout_shadows, (7,7), cv.BORDER_DEFAULT)
result = find_ill_parts(gauss)
fig.add_subplot(rows, columns, 1)
plt.imshow(result)
plt.title("Gaussian")

bilateral = cv.bilateralFilter(wihout_shadows, 15, 120, 75)
result = find_ill_parts(bilateral)
fig.add_subplot(rows, columns, 2)
plt.imshow(result)
plt.title("Bilateral")

non_local = cv.fastNlMeansDenoisingColored(img, None, 15, 15, 7, 21)
result = find_ill_parts(non_local)
fig.add_subplot(rows, columns, 3)
plt.imshow(result)
plt.title("Non-Local means")

plt.show()

