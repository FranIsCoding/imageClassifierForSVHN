import numpy as np
import matplotlib.pyplot as plt


class DataProcessor:
    def __init__(self):
        self.__train_data = None
        self.__test_data = None

    @property
    def training_data(self):
        return self.__train_data

    @training_data.setter
    def training_data(self, new_data):
        self.__train_data = new_data

    @property
    def testing_data(self):
        return self.__test_data

    @testing_data.setter
    def testing_data(self, new_data):
        self.__test_data = new_data

    def print_sample(self):
        if self.__train_data is None:
            raise Exception("Data has not been set.")
        fig, axs = plt.subplots(2, 5)
        for j in [0, 1]:
            for i in range(5):
                index = np.random.randint(100, size=10)[i + (5 * j)]
                image = self.__train_data["X"][:, :, :, index]
                label = self.__train_data["y"][index]
                axs[j, i].imshow(image, cmap="gray")
                axs[j, i].axis("off")
                axs[j, i].set_title(f'Image {index},\n label {label}')
        plt.suptitle("Random Image Sample")
        plt.show()

    def print_sample_grey(self):
        if self.__train_data is None:
            raise Exception("Data has not been set.")
        fig, axs = plt.subplots(2, 5)
        for j in [0, 1]:
            for i in range(5):
                index = np.random.randint(100, size=10)[i + (5 * j)]
                image = self.__train_data["X"][index, :, :, :]
                label = self.__train_data["y"][index]
                axs[j, i].imshow(image, cmap="gray")
                axs[j, i].axis("off")
                axs[j, i].set_title(f'Image {index},\n label {label}')
        plt.suptitle("Random Image Sample \n Grey Version")
        plt.show()

    def grey_scaling(self):
        if self.__train_data is None:
            raise Exception("Data has not been set.")
        self.__train_data["X"] = np.moveaxis(self.__train_data["X"], -1, 0)
        self.__train_data["X"] = np.mean(self.__train_data["X"], 3).reshape(73257, 32, 32, 1)/255
        self.__test_data["X"] = np.moveaxis(self.__test_data["X"], -1, 0)
        self.__test_data["X"] = np.mean(self.__test_data["X"], 3).reshape(26032, 32, 32, 1)/255
