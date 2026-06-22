import numpy as np
import tensorflow as tf
import tkinter as tk
from PIL import Image, ImageDraw, ImageTk
import matplotlib.cm as cm
from tensorflow.keras import models

model = tf.keras.models.load_model('model.keras')

conv_layers = [layer for layer in model.layers if 'conv2d' in layer.name or 'activation' in layer.name]

visualization_model = models.Model(
    inputs=model.inputs,
    outputs=[layer.output for layer in conv_layers]
)


# Функции для визуализации
def get_activations(img_array_28x28x1):
    """Прогоняет картинку через модель и возвращает активации всех слоев"""
    img = img_array_28x28x1.reshape(1, 28, 28, 1)
    activations = visualization_model.predict(img, verbose=0)
    return activations


def create_heatmap(activation_2d, size=(100, 100)):
    """Превращает двумерный массив активаций в цветную картинку"""
    # Нормализация
    act_min = activation_2d.min()
    act_max = activation_2d.max()
    if act_max == act_min:
        normalized = activation_2d
    else:
        normalized = (activation_2d - act_min) / (act_max - act_min)

    # Раскрашиваем
    colored = cm.jet(normalized)
    colored_rgb = (colored[:, :, :3] * 255).astype(np.uint8)

    img = Image.fromarray(colored_rgb, mode='RGB')
    img = img.resize(size, Image.NEAREST)
    return ImageTk.PhotoImage(img)


def create_grayscale_image(img_array_28x28, size=(100, 100)):
    """Превращает черно-белый массив 28x28 в картинку"""
    img_data = (img_array_28x28.squeeze() * 255).astype(np.uint8)
    img = Image.fromarray(img_data, mode='L')
    img = img.resize(size, Image.NEAREST)
    return ImageTk.PhotoImage(img)


def update_visualization(img_array_28x28):
    """Обновляет все три ряда картинок."""
    # 1: Обработанное изображение
    photo = create_grayscale_image(img_array_28x28)
    processed_label.configure(image=photo)
    processed_label.image = photo

    # 2: Карты активации последнего слоя
    activations = get_activations(img_array_28x28)

    last_activation = activations[-1][0]  # (1, H, W, фильтры) -> (H, W, фильтры)
    num_filters = last_activation.shape[-1]  # 64 фильтров

    for i in range(8):
        if i < num_filters:
            heatmap = last_activation[:, :, i]
            photo = create_heatmap(heatmap)
        else:
            # Если фильтров меньше 8 — пустая серая заглушка
            empty = np.ones((100, 100, 3), dtype=np.uint8) * 200
            photo = ImageTk.PhotoImage(Image.fromarray(empty))
        activation_labels[i].configure(image=photo)
        activation_labels[i].image = photo

    # 3: Послойная визуализация
    for i, act in enumerate(activations):
        avg_activation = np.mean(act[0], axis=-1)
        photo = create_heatmap(avg_activation)
        layer_labels[i].configure(image=photo)
        layer_labels[i].image = photo


# ОКНО
window = tk.Tk()
window.title("Digit recognizer")
window.geometry("800x650")

# Главный контейнер: слева рисовалка, справа визуализация
main_frame = tk.Frame(window)
main_frame.pack(padx=10, pady=10)

# Левая панель — рисовалка
left_panel = tk.Frame(main_frame)
left_panel.pack(side="left", padx=(0, 10))

canvas = tk.Canvas(
    left_panel,
    height=280,
    width=280,
    bg="white",
)
canvas.pack(pady=5)

image = Image.new("L", (280, 280), "white")
draw_image = ImageDraw.Draw(image)


def draw(event):
    x, y = event.x, event.y
    canvas.create_oval(x - 8, y - 8, x + 8, y + 8, fill="black")
    draw_image.ellipse((x - 8, y - 8, x + 8, y + 8), fill="black")


canvas.bind("<B1-Motion>", draw)


def clear_canvas():
    canvas.delete("all")
    global image, draw_image
    image = Image.new("L", (280, 280), "white")
    draw_image = ImageDraw.Draw(image)
    result_label.config(text="Нарисуй цифру")


def predict():
    # Обработка изображения
    img = image.resize((28, 28))
    img_array = np.array(img)
    img_array = (255 - img_array) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    # Предсказание
    prediction = model.predict(img_array, verbose=0)
    digit = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    result_label.config(text=f"Цифра: {digit} ({confidence:.1f}%)")

    # Визуализация
    update_visualization(img_array[0])


# Кнопки
predict_button = tk.Button(left_panel, text="Распознать", command=predict)
predict_button.pack(pady=5)

clear_button = tk.Button(left_panel, text="Очистить", command=clear_canvas)
clear_button.pack(pady=5)

result_label = tk.Label(left_panel, text="Нарисуй цифру", font=("Arial", 16))
result_label.pack(pady=5)

# Правая панель — визуализация
right_panel = tk.Frame(main_frame)
right_panel.pack(side="left")

# Ряд 1: Обработанное изображение
tk.Label(right_panel, text="Что видит нейросеть (28×28)",
         font=('Arial', 9, 'bold')).pack(pady=(0, 2))

processed_label = tk.Label(right_panel, bg='white', width=100, height=100)
processed_label.pack(pady=(0, 10))

# Ряд 2: Карты активации
tk.Label(right_panel, text="Карты активации (первые 8 фильтров последнего слоя)",
         font=('Arial', 9, 'bold')).pack(pady=(0, 2))

activation_frame = tk.Frame(right_panel)
activation_frame.pack(pady=(0, 10))

activation_labels = []
for i in range(8):
    label = tk.Label(activation_frame, bg='white', width=100, height=100)
    label.pack(side="left", padx=1)
    activation_labels.append(label)

# Ряд 3: Послойная визуализация
tk.Label(right_panel, text="Послойная визуализация (усреднение по фильтрам)",
         font=('Arial', 9, 'bold')).pack(pady=(0, 2))

layer_frame = tk.Frame(right_panel)
layer_frame.pack(pady=(0, 5))

layer_labels = []
for i in range(8):
    label = tk.Label(layer_frame, bg='white', width=100, height=100)
    label.pack(side="left", padx=1)
    layer_labels.append(label)

# Подписи к слоям
layer_names = []
for layer in conv_layers:
    name = layer.name
    if 'conv2d' in name:
        name = name.replace('conv2d', 'C').replace('_', '')
    elif 'activation' in name:
        name = name.replace('activation', 'R').replace('_', '')
    layer_names.append(name)

name_frame = tk.Frame(right_panel)
name_frame.pack()
for name in layer_names:
    tk.Label(name_frame, text=name, font=('Arial', 7), width=12).pack(side="left", padx=1)

window.mainloop()
