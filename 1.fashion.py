import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_data():
    ''' Загрузка данных '''
    
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()    

    return (train_images, train_labels), (test_images, test_labels)
    

def open_image(image, save=None):
    plt.figure()
    plt.imshow(image)
    plt.colorbar()
    plt.grid(False)
    if save is not None:
        plt.savefig(save)
    plt.show()


def result(predictions, x):
    max = predictions[x].max()
    min = predictions[x].min()
    
    if predictions[x][0] == max:
        text = f'clothes: {round(max*100, 2)}%'
    elif predictions[x][1] == max:
        text = f'shoes: {round(max*100, 2)}%'
    else: pass

    return text


if __name__=='__main__':
    # Загрузка данных

    (train_images, train_labels), (test_images, test_labels) = load_data()
    # train_images, train_labels - данные для обучения
    # test_images, test_labels - данные для теста

    # файлы _labels - массив от 0 до 9
    # они соответствуюют классу одежды:

    # 'T-shirt/top' = 0
    # 'Trouser' = 1
    # 'Pullover' = 2
    # 'Dress' = 3
    # 'Coat' = 4
    # 'Sandal' = 5
    # 'Shirt' = 6
    # 'Sneaker' = 7
    # 'Bag' = 8
    # 'Ankle boot' = 9

    # Но т.к. наша задача лишь в том, чтобы определить одежда это или обувь,
    # отредактируем файлы _labels

    # clothes, shoes = одежда, обувь

    # {
    # 'shoes':[5,7,9],
    # 'clothes':[0,1,2,3,4,6,8]
    # }

    item_type = ['clothes', 'shoes']

    train_labels = pd.DataFrame(train_labels, columns={'labels'})
    train_labels['labels2'] = train_labels['labels'].apply(lambda x: 1 if x in [5,7,9] else 0)
    train_labels = train_labels['labels2'].values
    
    test_labels = pd.DataFrame(test_labels, columns={'labels'})
    test_labels['labels2'] = test_labels['labels'].apply(lambda x: 1 if x in [5,7,9] else 0)
    test_labels = test_labels['labels2'].values

    # print (train_images.shape)    # (60000, 28, 28)
    # print (train_labels.shape)    # (60000,)
    # print (test_images.shape)     # (10000, 28, 28)
    # print (test_labels.shape)     # (10000,)
    
    # view and save file
    # open_image(train_images[1], 'files/1.image.png')

    # Нормализация цвета [0..1]
    test_images = test_images / 255
    train_images = train_images / 255

    # Проверка изображений
    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i], cmap=plt.cm.binary)
        plt.xlabel(item_type[train_labels[i]])
    plt.savefig('files/input.png')

    # Настройка слоев
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),         # преобразование массива из двумерного в одномерный
        keras.layers.Dense(128, activation=tf.nn.relu),     # связанные нейронные слои. 128 нейронов
        keras.layers.Dense(2, activation=tf.nn.softmax)     # вероятность класса
    ])

    model.compile(
        optimizer=tf.train.AdamOptimizer(), 
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )


    # Обучение:
    model.fit(train_images, train_labels, epochs=1)


    # Тест
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print('Test accuracy:', test_acc)


    # Прогнозирование:
    predictions = model.predict(test_images)
    
    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(test_images[i], cmap=plt.cm.binary)
        text = result(predictions, i)
        plt.xlabel(text)
    plt.savefig('files/result.png')
    