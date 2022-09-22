import gzip
import numpy as np
import matplotlib.pyplot as plt

import os


def read_labels(rel_path, lab_offset=8):
    file_path = os.path.abspath(rel_path)
    with gzip.open(file_path, "rb") as labels_file:
        labels = np.frombuffer(labels_file.read(), dtype=np.uint8, offset=lab_offset)
        return labels


def read_images(rel_path, length, resolution=28, img_offset=16):
    file_path = os.path.abspath(rel_path)
    with gzip.open(file_path, "rb") as images_file:
        images = np.frombuffer(images_file.read(), dtype=np.uint8, offset=img_offset).reshape(length, resolution**2)
        return images


def read_data(lab_path, img_path, resolution=28, img_offset=16, lab_offset=8):
    labels = read_labels(lab_path, lab_offset)
    images = read_images(img_path, length=len(labels), resolution=resolution, img_offset=img_offset)
    return labels, images


def shuffle_data(labels, images):
    indices = np.arange(labels.shape[0])
    np.random.shuffle(indices)

    shuffled_labels = labels[indices]
    shuffled_images = images[indices]
    return shuffled_labels, shuffled_images


if __name__ == "__main__":
    test_labels_path = "data/fashion/t10k-labels-idx1-ubyte.gz"
    train_labels_path = "data/fashion/train-labels-idx1-ubyte.gz"

    test_images_path = "data/fashion/t10k-images-idx3-ubyte.gz"
    train_images_path = "data/fashion/train-images-idx3-ubyte.gz"

    image_resolution = 28
    test_labels, test_images = read_data(test_labels_path, test_images_path)
    shuffled_labels, shuffled_images = shuffle_data(test_labels, test_images)
  
    for image_index in range(0, 9):
        plt.subplot(330 + 1 + image_index)
        plt.imshow(shuffled_images[image_index].reshape(image_resolution, image_resolution), cmap=plt.get_cmap("gray"))
    plt.show()