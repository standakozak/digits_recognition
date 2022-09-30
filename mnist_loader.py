import gzip
import numpy as np
import matplotlib.pyplot as plt

import os


def load_images(rel_path, length, resolution=28, img_offset=16):
    file_path = os.path.abspath(rel_path)
    with gzip.open(file_path, "rb") as images_file:
        images = np.frombuffer(images_file.read(), dtype=np.uint8, offset=img_offset).reshape(length, resolution**2, 1)
        return np.asarray(images/255)


def load_labels(rel_path, lab_offset=8):
    file_path = os.path.abspath(rel_path)
    with gzip.open(file_path, "rb") as labels_file:
        labels = np.frombuffer(labels_file.read(), dtype=np.uint8, offset=lab_offset)
        return np.asarray(labels)


def load_data(img_path, lab_path, resolution=28, img_offset=16, lab_offset=8):
    labels = load_labels(lab_path, lab_offset)
    images = load_images(img_path, length=len(labels), resolution=resolution, img_offset=img_offset)
    return list(zip(images, labels))


def shuffle_data(data):
    new_data = data.copy()
    np.random.shuffle(new_data)
    return new_data


def vectorize(data):
    vectorized_train_data = []
    for input, output_number in data:
        new_outputs = np.zeros((10, 1))
        new_outputs[output_number] = 1.0
        vectorized_train_data.append((input, new_outputs))
    return vectorized_train_data


def split_validation_train_data(all_train_data, validation_data_num):
    if validation_data_num:
        validation_data = (all_train_data[-validation_data_num:])
        train_data = (all_train_data[:-validation_data_num])
    else:
        validation_data = []
        train_data = all_train_data
    return train_data, validation_data


def load_mnist(validation_data_num=10000):
    test_images_path = "data/mnist/t10k-images-idx3-ubyte.gz"
    test_labels_path = "data/mnist/t10k-labels-idx1-ubyte.gz"
    test_data = load_data(test_images_path, test_labels_path)

    train_images_path = "data/mnist/train-images-idx3-ubyte.gz"
    train_labels_path = "data/mnist/train-labels-idx1-ubyte.gz"
    all_train_data = load_data(train_images_path, train_labels_path)
    train_data, validation_data = split_validation_train_data(all_train_data, validation_data_num)
    
    return (vectorize(train_data), vectorize(validation_data), vectorize(test_data))

def load_fashion(validation_data_num=10000):
    test_images_path = "data/fashion/t10k-images-idx3-ubyte.gz"
    test_labels_path = "data/fashion/t10k-labels-idx1-ubyte.gz"    
    test_data = load_data(test_images_path, test_labels_path)

    train_images_path = "data/fashion/train-images-idx3-ubyte.gz"
    train_labels_path = "data/fashion/train-labels-idx1-ubyte.gz"
    all_train_data = load_data(train_images_path, train_labels_path)
    train_data, validation_data = split_validation_train_data(all_train_data, validation_data_num)

    return (vectorize(train_data), vectorize(validation_data), vectorize(test_data))


if __name__ == "__main__":
    image_resolution = 28
    train_data, validation_data, test_data = load_mnist()
    shuffled_test = shuffle_data(test_data)
  
    for image_index in range(0, 9):
        plt.subplot(330 + 1 + image_index)
        plt.imshow(shuffled_test[image_index][0].reshape(image_resolution, image_resolution), cmap=plt.get_cmap("gray"))
    plt.show()