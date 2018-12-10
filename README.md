# Definition_of_clothes_and_shoes_in_the_image

Data:

https://github.com/zalandoresearch/fashion-mnist

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()   

train_images, train_labels - данные для обучения

test_images, test_labels - данные для теста

файлы _labels - массив от 0 до 9

они соответствуюют классу одежды:

'T-shirt/top' = 0

'Trouser' = 1

'Pullover' = 2

'Dress' = 3

'Coat' = 4

'Sandal' = 5

'Shirt' = 6

'Sneaker' = 7

'Bag' = 8

'Ankle boot' = 9

Но т.к. наша задача лишь в том, чтобы определить одежда это или обувь,

отредактируем файлы _labels

clothes, shoes = одежда, обувь

{
'shoes':[5,7,9],

'clothes':[0,1,2,3,4,6,8]
}
