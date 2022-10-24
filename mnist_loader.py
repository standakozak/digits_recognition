import gzip
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import quickdraw
import os

import cProfile


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

def shuffle_data(data):
    new_data = data.copy()
    np.random.shuffle(new_data)
    return new_data

def resize_image(image_arr, orig_size=256, desired_size=28):
    return np.array(Image.fromarray(image_arr).resize((desired_size, desired_size)))
    # step = int(orig_size / desired_size)
    # resized = []
    # for row_index in range(desired_size):
    #     row = row_index * step
    #     for col_index in range(desired_size):
    #         col = col_index*step
    #         resized.append((image_arr[row:row+step, col:col+step]).mean())

    # resized_arr = np.asarray(resized)
    # return resized_arr

def quickdraw_category_path(category):
    return f"data/quick_draw/{category}.npy"

def check_saved(category):
    file_name = quickdraw_category_path(category)
    return os.path.exists(file_name)

def load_doodles_from_file(category, category_number):
    file_name = quickdraw_category_path(category)
    
    data = np.load(file_name, allow_pickle=True)
    data_to_return = [(np.asarray(image).reshape(784, 1), category_number) for image in data]

    return data_to_return

def save_doodles_category(data, category):
    file_name = quickdraw_category_path(category)
    data_to_save = [image[0].tolist() for image in data]
    np.save(file_name, data_to_save, allow_pickle=True)

def load_new_doodles_category(category, num_of_drawings, category_number):
    data = []
    data_group = quickdraw.QuickDrawDataGroup(category, recognized=True, max_drawings=num_of_drawings)
    for quick_draw_image in data_group.drawings:
        resized_image = quick_draw_image.image.resize((28, 28))
        resized = 1 - (np.asarray(resized_image).mean(axis=2) / 255)
        # image_arr = 1 - (np.asarray(quick_draw_image.image).mean(axis=2) / 255)

        # resized = resize_image(image_arr, 256, 28)
        data.append((resized.reshape(784, 1), category_number))
    save_doodles_category(data, category)
    return data

def load_doodles(categories_list=None):
    total_data_length = 70_000
    totaL_test_length = 10_000
    total_validation_length = 10_000
    
    if categories_list is None:
        categories = [
            "axe", "bicycle", "broom", "bucket", "candle", "chair", "eyeglasses", "guitar", "key", "ladder"
        ]
    else:
        categories = categories_list.copy()

    num_of_each_category = total_data_length / len(categories)
    data = []
    for category_index, category in enumerate(categories):
        if check_saved(category):
            data += load_doodles_from_file(category, category_index)
        else:
            data += load_new_doodles_category(category, num_of_each_category, category_index)

    totaL_train_length = total_data_length - totaL_test_length - total_validation_length

    vectorized = vectorize(shuffle_data(data))
    train_data = vectorized[:totaL_train_length]
    test_data = vectorized[totaL_train_length:totaL_train_length+totaL_test_length]
    validation_data = vectorized[totaL_train_length + totaL_test_length:]

    return (train_data, validation_data, test_data)

def load_fashion(validation_data_num=10000):
    test_images_path = "data/fashion/t10k-images-idx3-ubyte.gz"
    test_labels_path = "data/fashion/t10k-labels-idx1-ubyte.gz"    
    test_data = load_data(test_images_path, test_labels_path)

    train_images_path = "data/fashion/train-images-idx3-ubyte.gz"
    train_labels_path = "data/fashion/train-labels-idx1-ubyte.gz"
    all_train_data = load_data(train_images_path, train_labels_path)
    train_data, validation_data = split_validation_train_data(all_train_data, validation_data_num)

    return (vectorize(train_data), vectorize(validation_data), vectorize(test_data))


def load_mnist(validation_data_num=10000):
    test_images_path = "data/mnist/t10k-images-idx3-ubyte.gz"
    test_labels_path = "data/mnist/t10k-labels-idx1-ubyte.gz"
    test_data = load_data(test_images_path, test_labels_path)

    train_images_path = "data/mnist/train-images-idx3-ubyte.gz"
    train_labels_path = "data/mnist/train-labels-idx1-ubyte.gz"
    all_train_data = load_data(train_images_path, train_labels_path)
    train_data, validation_data = split_validation_train_data(all_train_data, validation_data_num)
    
    return (vectorize(train_data), vectorize(validation_data), vectorize(test_data))


def show_plot_images(data):
    image_resolution = 28

    shuffled_data = shuffle_data(data)
    for image_index in range(0, 9):
        plt.subplot(330+image_index+1)
        plt.imshow(shuffled_data[image_index][0].reshape(image_resolution, image_resolution), cmap=plt.get_cmap("gray"))
    plt.show()


if __name__ == "__main__":
    with cProfile.Profile() as pr:
        _, _, test_data_doodles = load_doodles()
        pr.print_stats()

    _, _, test_data_mnist = load_mnist()
    show_plot_images(test_data_doodles)
    show_plot_images(test_data_mnist)
