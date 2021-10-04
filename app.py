import os

import numpy as np
from skimage.transform import resize
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
from matplotlib import pyplot as plt

import torch
from torchvision import datasets, transforms

from mnist import Net

model_path = "mnist_cnn.pt"


def load_model():
    model = Net()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


tfm = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((28, 28), antialias=True),
        transforms.Normalize((0.1307,), (0.3081,)),
    ]
)

model = load_model()

canvasH = 28 * 20
canvasW = 28 * 20


def app1():
    st.markdown(
        """
        # 手書き認識
        - write a digit from 0 to 9
    """
    )

    # adjust layout
    _, col = st.columns((0.1, 0.8))
    with col:
        canvas = st_canvas(
            stroke_width=28 * 2,
            stroke_color="black",
            background_image=Image.fromarray(
                255 * np.ones((canvasH, canvasW)).astype(np.uint8)
            ),
            height=canvasH,
            width=canvasW,
            key="app1",
        )

    user_wrote_something = np.sum(canvas.image_data)
    if user_wrote_something:
        gray = np.sum(canvas.image_data, axis=2).astype(
            np.uint8
        )  # warning: must cast to uint8
        output = model(torch.unsqueeze(tfm(gray), dim=0)).detach().squeeze().numpy()
        prob = 100 * np.exp(output)
        label = np.argmax(prob)

        st.markdown("# 予測結果 {}".format(label))

        fig, (ax1, ax2) = plt.subplots(2, 1)

        ax1.set_title("input image")
        ax1.imshow(tfm(gray).squeeze(), cmap="gray")

        ax2.bar(range(10), prob)
        ax2.set_xlabel("digit")
        ax2.set_xticks(range(10))
        ax2.set_xticklabels(range(10))
        ax2.set_ylabel("prob")
        ax2.set_yticks(10 * np.arange(10))
        ax2.set_yticklabels(["{}".format(10 * p) for p in range(10)])

        st.pyplot(fig)


PAGES = {
    "app1": app1,
}


def main():
    app1()


if __name__ == "__main__":
    main()
