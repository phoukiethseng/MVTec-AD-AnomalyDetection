import cv2 as cv
import matplotlib.pyplot as plt
from setuptools.sandbox import save_path

files = [
    "./dataset/bottle/test/broken_large/005.png",
    "./dataset/cable/test/bent_wire/006.png",
    "./dataset/transistor/test/damaged_case/003.png",
    "./dataset/metal_nut/test/scratch/008.png"
]

category = [
    "bottle",
    "cable",
    "transistor",
    "metal_nut"
]

for i, f in enumerate(files):
    img = cv.imread(f)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    grad_x = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=3)
    grad_y = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=3)

    # grad = np.sqrt(grad_x**2 + grad_y**2)
    abs_grad_x = cv.convertScaleAbs(grad_x)
    abs_grad_y = cv.convertScaleAbs(grad_y)

    grad = cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    save_path = f"../output/{category[i]}_grad.png"
    cv.imwrite(save_path, grad)
cv.waitKey(0)