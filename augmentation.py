import glob
import numpy as np
import telegram
import tensorflow as tf
from tqdm import tqdm
from scipy import ndimage
import matplotlib.pyplot as plt
import glob
import cv2 as cv
import random
from tqdm import tqdm
import numpy as np
from PIL import Image
from multiprocessing.pool import ThreadPool


class Imageloader:
    def __init__(
            self, total_class_num, img_size, ang, zoom, xtrans, ytrans, alpha, beta, seed, noise_ratio, flip=True,
            apply_clahe=True
    ):
        self.ang = ang
        self.zoom = zoom
        self.xtrans = xtrans
        self.ytrans = ytrans
        self.alpha = alpha
        self.beta = beta
        self.total_class_num = total_class_num
        self.img_size = img_size
        self.apply_clahe = apply_clahe
        self.noise_ratio = noise_ratio
        self.flip = flip
        random.seed(seed)

    def apply_CLAHE(self, image):
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image = clahe.apply(image)
        image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
        return image

    def apply_flip(self, image):
        image = cv.flip(image, 1)
        return image

    def apply_transform(self, image):

        ang = self.ang * random.uniform(-1, 1)
        zoom = (self.zoom * random.uniform(-1, 1)) + 1
        xtrans = self.xtrans * random.uniform(-1, 1)
        ytrans = self.ytrans * random.uniform(-1, 1)
        alpha = (self.alpha * random.uniform(-1, 1)) + 1
        beta = self.beta * random.uniform(-1, 1)

        height, width, channel = image.shape

        matrix = cv.getRotationMatrix2D((width / 2, height / 2), ang, zoom)
        image = cv.warpAffine(image, matrix, (width, height))
        matrix = np.float32([[1, 0, width * xtrans], [0, 1, height * ytrans]])
        image = cv.warpAffine(image, matrix, (width, height))
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        image = image.astype("float32")
        image = np.clip(image * alpha + beta, 0, 255)
        image = image.astype("uint8")
        image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)

        return image

    def resize_image(self, img):

        size = (self.img_size, self.img_size)
        h, w = img.shape[:2]
        c = img.shape[2] if len(img.shape) > 2 else 1
        if h == w:
            return cv.resize(img, size, cv.INTER_AREA)
        dif = h if h > w else w

        interpolation = cv.INTER_AREA

        x_pos = (dif - w) // 2
        y_pos = (dif - h) // 2

        if len(img.shape) == 2:
            mask = np.zeros((dif, dif), dtype=img.dtype)
            mask[y_pos: y_pos + h, x_pos: x_pos + w] = img[:h, :w]
        else:
            mask = np.zeros((dif, dif, c), dtype=img.dtype)
            mask[y_pos: y_pos + h, x_pos: x_pos + w, :] = img[:h, :w, :]

        return cv.resize(mask, size, interpolation)

    def addsalt_pepper(self, img):
        img_ = img.copy()
        SNR = max(min((self.noise_ratio), 1), 0)
        h, w, _ = img_.shape
        num_salt = h * w * SNR / 2

        # Add Salt noise
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img_.shape]
        img_[coords[0], coords[1], :] = 255

        # Add Pepper noise
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img_.shape]
        img_[coords[0], coords[1], :] = 0

        return img_

    def loadimg(self, filepath):
        return cv.imread(filepath)

    def do_transform(self, img):
        if (random.randint(0, 1) == 1) & self.apply_clahe:
            img = self.apply_CLAHE(img)
        img = self.apply_transform(img)
        img = self.resize_image(img)
        if (self.noise_ratio > 0) & (random.randint(0, 1) == 1):
            img = self.addsalt_pepper(img)
        if (random.randint(0, 1) == 1) & self.flip:
            img = self.apply_flip(img)
        return img

    def augmented_loader(self, file_list, ratio):

        pool = ThreadPool()
        imgs = list(tqdm(pool.imap(self.loadimg, file_list), total=len(file_list), desc="Loading files    "))
        imgs_org = pool.map(self.resize_image, imgs)
        imgs1 = imgs * (ratio - 1)
        imgs2 = list(tqdm(pool.imap(self.do_transform, imgs1), total=len(imgs1), desc="Augmenting images"))
        imgs_org += imgs2
        pool.close()

        return [x / 255.0 for x in imgs_org]

    def imageset_load_aug(self, file_list, ratio, class_index, class_num):
        classes = []
        imgs = np.array(self.augmented_loader(file_list, ratio))
        for i in range(len(imgs)):
            classes.append(class_index)
        classes = np.eye(class_num)[classes]
        return imgs, classes

    def non_augmented_loader(self, file_list):
        pool = ThreadPool()
        imgs = list(tqdm(pool.imap(self.loadimg, file_list), total=len(file_list), desc="Loading files    "))
        imgs_org = pool.map(self.resize_image, imgs)
        pool.close()

        return [x / 255.0 for x in imgs_org]

    def imageset_load_nonaug(self, file_list, class_index, class_num):
        classes = []
        imgs = np.array(self.non_augmented_loader(file_list))
        for i in range(len(imgs)):
            classes.append(class_index)
        classes = np.eye(class_num)[classes]
        return imgs, classes

    def clahe_loader(self, file_list):
        pool = ThreadPool()
        imgs = list(tqdm(pool.imap(self.loadimg, file_list), total=len(file_list), desc="Loading files    "))
        imgs = pool.map(self.apply_CLAHE, imgs)
        imgs_org = pool.map(self.resize_image, imgs)
        pool.close()

        return [x / 255.0 for x in imgs_org]

    def imageset_load_clahe(self, file_list, class_index, class_num):
        classes = []
        imgs = np.array(self.clahe_loader(file_list))
        for i in range(len(imgs)):
            classes.append(class_index)
        classes = np.eye(class_num)[classes]
        return imgs, classes


def get_gradcam(model, IMAGE_PATH, CLASS_INDEX, LAYER_NAME):
    img = tf.keras.preprocessing.image.load_img(IMAGE_PATH, target_size=(224, 224))
    image_array = np.array(img)
    img = image_array.astype(np.float32) / 255.0

    # Create a graph that outputs target convolution and output
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(LAYER_NAME).output, model.output])

    # Get the score for target class
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(np.array([img]))
        loss = predictions[:, CLASS_INDEX]

    # Extract filters and gradients
    output = conv_outputs[0]
    grads = tape.gradient(loss, conv_outputs)[0]

    # Average gradients spatially
    weights = tf.reduce_mean(grads, axis=(0, 1))

    # Build a ponderated map of filters according to gradients importance
    cam = np.ones(output.shape[0:2], dtype=np.float32)

    for index, w in enumerate(weights):
        cam += w * output[:, :, index]

    # Heatmap visualization
    cam = cv.resize(cam.numpy(), (224, 224))
    cam = np.maximum(cam, 0)

    heatmap = (cam - cam.min()) / (cam.max() - cam.min())
    heatmap *= heatmap

    cam = cv.applyColorMap(np.uint8(255 * heatmap), cv.COLORMAP_JET)

    alphanum = 0.6

    output_image = cv.addWeighted(
        cv.cvtColor(image_array.astype("uint8"), cv.COLOR_RGB2BGR), alphanum, cam, 1 - alphanum, 0,
    )
    output_image = cv.cvtColor(output_image.astype("uint8"), cv.COLOR_BGR2RGB)

    return output_image


def get_class(model, IMAGE_PATH):
    img = tf.keras.preprocessing.image.load_img(IMAGE_PATH, target_size=(224, 224))
    image_array = np.array(img)
    img = image_array.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    predictions = model.predict(img)
    y_pred = np.argmax(predictions, axis=1)
    return y_pred


def show_gradcam(model, LAYER_NAME, IMG_PATH, CLASS_INDEX):
    FILE_LIST = glob.glob(IMG_PATH)
    FILE_LIST2 = []
    for IMAGE_PATH in glob.glob(IMG_PATH):
        predictions = get_class(model, IMAGE_PATH)
        if predictions[0] == CLASS_INDEX:
            FILE_LIST2.append(IMAGE_PATH)

    x_num = 5
    y_num = round(len(FILE_LIST2) / x_num + 0.5)
    i = 1
    fig = plt.figure(figsize=(x_num * 4, y_num * 4))

    for IMAGE_PATH in FILE_LIST2:
        plt.subplot(y_num, x_num, i)
        output_image = get_gradcam(model, IMAGE_PATH, CLASS_INDEX, LAYER_NAME).astype("uint8")
        setting = plt.gca()
        setting.axes.xaxis.set_visible(False)
        setting.axes.yaxis.set_visible(False)
        plt.imshow(output_image)
        i = i + 1


def save_gradcam(model, LAYER_NAME, IMG_PATH, CLASS_INDEX, SAVE_PATH):
    FILE_LIST = glob.glob(IMG_PATH)
    FILE_LIST2 = []
    for IMAGE_PATH in glob.glob(IMG_PATH):
        predictions = get_class(model, IMAGE_PATH)
        if predictions[0] == CLASS_INDEX:
            FILE_LIST2.append(IMAGE_PATH)

    i = 1
    fig = plt.figure(figsize=(5, 5))

    for IMAGE_PATH in FILE_LIST2:
        output_image = get_gradcam(model, IMAGE_PATH, CLASS_INDEX, LAYER_NAME).astype("uint8")
        setting = plt.gca()
        setting.axes.xaxis.set_visible(False)
        setting.axes.yaxis.set_visible(False)
        filename = SAVE_PATH + str(i) + ".jpg"
        plt.imshow(output_image)
        plt.savefig(filename, dpi=300)
        i = i + 1