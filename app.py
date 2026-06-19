import numpy as np
import tensorflow as tf
import tkinter as tk
from PIL import Image, ImageDraw

model = tf.keras.models.load_model('model.keras')

window = tk.Tk()
window.title("Digit recognizer")
window.geometry("400x400")

canvas = tk.Canvas(
    window,
    height=280,
    width=280,
    bg="white",
)
canvas.pack(pady=10)

image = Image.new("L", (280, 280), "white")
draw_image = ImageDraw.Draw(image)


def draw(event):
    x, y = event.x, event.y

    canvas.create_oval(
        x - 8,
        y - 8,
        x + 8,
        y + 8,
        fill="black",
    )

    draw_image.ellipse(
        (x - 8, y - 8, x + 8, y + 8),
        fill="black",
    )


canvas.bind("<B1-Motion>", draw)


def clear_canvas():
    canvas.delete("all")

    global image, draw_image

    image = Image.new("L", (280, 280), "white")
    draw_image = ImageDraw.Draw(image)

    result_label.config(text="Нарисуй цифру")


def predict():
    img = image.resize((28, 28))
    img_array = np.array(img)

    img_array = 255 - img_array
    img_array = img_array / 255.0

    img_array = img_array.reshape(1, 28, 28, 1)
    prediction = model.predict(img_array)

    digit = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    result_label.config(
        text=f"Цифра: {digit} ({confidence:.1f}%)"
    )


predict_button = tk.Button(
    window,
    text="Распознать",
    command=predict
)
predict_button.pack(pady=5)

clear_button = tk.Button(
    window,
    text="Отчистить",
    command=clear_canvas
)
clear_button.pack(pady=10)

result_label = tk.Label(
    window,
    text="Нарисуй цифру",
    font=("Arial", 16)
)
result_label.pack(pady=10)

window.mainloop()
