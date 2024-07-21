from scipy.io import loadmat
from dataProcessor import DataProcessor
from nnClassifiers import MlpClassifier, ConvClassifier
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    processor = DataProcessor()
    train = loadmat('data/train_32x32.mat')
    test = loadmat('data/test_32x32.mat')
    predict = test.copy()
    processor.training_data = train
    processor.testing_data = test
    processor.grey_scaling()
    train_data = processor.training_data
    test_data = processor.testing_data

    classfier1 = ConvClassifier()
    classfier1.training_data = train_data
    classfier1.testing_data = test_data
    classfier1.model_generator()
    classfier1.model_summary()
    classfier1.model_train()
    classfier1.train_statistics()

    mlpModel = load_model("best_weight_model_mlp/best_model.keras")
    cnnModel = load_model("best_weight_model_cnn/best_model.keras")
