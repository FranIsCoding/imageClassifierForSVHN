from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt


class MlpClassifier:
    def __init__(self):
        self.train_data = None
        self.test_data = None
        self.model = None
        self.history = None
        self.callbacks = None

    @property
    def training_data(self):
        return self.train_data

    @training_data.setter
    def training_data(self, new_data):
        self.train_data = new_data

    @property
    def testing_data(self):
        return self.test_data

    @testing_data.setter
    def testing_data(self, new_data):
        self.test_data = new_data

    def model_summary(self):
        if self.model is None:
            raise Exception("No model defined")
        self.model.summary()

    def model_generator(self):
        model = Sequential([
            Flatten(input_shape=(32, 32, 1)),
            Dense(1024, activation='relu'),
            Dense(512, activation='relu'),
            Dense(256, activation='relu'),
            Dense(64, activation='relu'),
            Dense(11, activation='softmax'),
        ])
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        self.model = model

    def generate_callbacks(self):
        checkpoint = ModelCheckpoint(filepath='best_weight_model_mlp/best_model.keras',
                                     save_best_only=True,
                                     monitor="val_accuracy",
                                     mode="max")

        earlystop = EarlyStopping(monitor="val_accuracy",
                                  patience=10)

        self.callbacks = [earlystop, checkpoint]

    def model_train(self):
        if self.train_data is None:
            raise Exception("Data has not been set.")
        self.generate_callbacks()
        self.history = self.model.fit(self.train_data['X'], self.train_data['y'],
                                      batch_size=256,
                                      epochs=20,
                                      validation_split=0.1,
                                      callbacks=self.callbacks)

    def train_statistics(self):
        train_acc = self.history.history["accuracy"]
        val_acc = self.history.history["val_accuracy"]
        train_loss = self.history.history["loss"]
        val_loss = self.history.history["val_loss"]
        epochs = range(1, len(train_loss) + 1)

        fig, axs = plt.subplots(1, 2)

        axs[0].plot(epochs, train_loss, 'r-', label='Training Loss')
        axs[0].plot(epochs, val_loss, 'b-', label='Validation Loss')
        axs[0].set_title('Training and Validation Loss')
        axs[0].set_xlabel('Epochs')
        axs[0].set_ylabel('Loss')
        axs[0].legend()

        axs[1].plot(epochs, train_acc, 'r-', label='Training Accuracy')
        axs[1].plot(epochs, val_acc, 'b-', label='Validation Accuracy')
        axs[1].set_title('Training and Validation Accuracy')
        axs[1].set_xlabel('Epochs')
        axs[1].set_ylabel('Loss')
        axs[1].legend()

        plt.show()

    def test_statistics(self):
        test_loss, test_accuracy = self.model.evaluate(self.test_data['X'], self.test_data['y'])
        return test_loss, test_accuracy


class ConvClassifier(MlpClassifier):

    def model_generator(self):
        model = Sequential([
            Conv2D(16, (3, 3), activation='relu', input_shape=(32, 32, 1)),
            MaxPooling2D((3, 3)),
            Conv2D(16, (3, 3), activation='relu'),
            Flatten(),
            BatchNormalization(),
            Dense(64, activation='relu'),
            Dropout(0.1),
            Dense(32, activation='relu'),
            Dropout(0.1),
            Dense(11, activation='softmax')
        ])
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        self.model = model

    def generate_callbacks(self):
        checkpoint = ModelCheckpoint(filepath='best_weight_model_cnn/best_model.keras',
                                     save_best_only=True,
                                     monitor="val_accuracy",
                                     mode="max")

        earlystop = EarlyStopping(monitor="val_accuracy",
                                  patience=10)

        self.callbacks = [earlystop, checkpoint]
