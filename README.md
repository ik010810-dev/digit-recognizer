# Digit recognizer

Нейросеть для распознавания рукописных цифр (MNIST).

## Архитектура

- 2 сверточных блока (Conv2D + BatchNorm + ReLU)
- GlobalAveragePooling
- Полносвязный классификатор с Dropout
- Аугментация данных (повороты, сдвиги, zoom)

## Установка

```bash
pip install -r requirements.txt