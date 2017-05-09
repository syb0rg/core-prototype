import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
import tflearn

def main():
    LABELED_DIR = 'labeled_data'
    width = 512
    height = 512
    classes = 26  # characters
    learning_rate = 0.0001
    batch_size = 25

    # load data
    print('Loading data')
    X, Y = tflearn.data_utils.image_preloader(LABELED_DIR, image_shape=(width, height), mode='folder', normalize=True, grayscale=True, categorical_labels=True, files_extension=None, filter_channel=False)
    X_shaped = np.squeeze(X)
    trainX, trainY = X_shaped, Y

    # Network building
    print('Building network')
    net = tflearn.input_data(shape=[None, width, height])
    net = tflearn.lstm(net, 128, dropout=0.8)
    net = tflearn.fully_connected(net, classes, activation='softmax')
    net = tflearn.regression(net, optimizer='adam', learning_rate=learning_rate, loss='categorical_crossentropy')

    model = tflearn.DNN(net, tensorboard_verbose=3)
    print('Training network')
    model.fit(trainX, trainY, validation_set=0.15, n_epoch=100, show_metric=True, batch_size=batch_size)
    model.save("tflearn.lstm.model")


if __name__ == '__main__':
    main()
