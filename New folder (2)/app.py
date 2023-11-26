
import os
import streamlit as st
import tensorflow as tf
import numpy as np
import nibabel as nib
from scipy import ndimage
from scipy.ndimage import zoom
import cv2
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import os
import zipfile
import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
import cv2
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import tensorflow as tf
from ipywidgets import IntSlider, interact
from matplotlib import animation, rc
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.path import Path
from scipy import ndimage
from scipy.ndimage import zoom
from tensorflow.keras import Input, Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import (BatchNormalization, Conv3D, Dense,
                                     Dropout, GlobalAveragePooling3D,
                                     MaxPool3D)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay




# Function to read and preprocess the image
def read_nifti_file(filepath):
    scan = nib.load(filepath)
    scan = scan.get_fdata()
    return scan

def normalize(volume):
    min_val = -1000
    max_val = 400
    volume[volume < min_val] = min_val
    volume[volume > max_val] = max_val
    volume = (volume - min_val) / (max_val - min_val)
    volume = volume.astype("float32")
    return volume

def resize_volume(img):
    desired_depth = 64
    desired_width = 128
    desired_height = 128
    current_depth = img.shape[-1]
    current_width = img.shape[0]
    current_height = img.shape[1]
    depth = current_depth / desired_depth
    width = current_width / desired_width
    height = current_height / desired_height
    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height
    img = np.rot90(img, 3)
    img = ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=1)
    return img

def preprocess_image(file_path):
    volume = read_nifti_file(file_path)
    volume = normalize(volume)
    volume = resize_volume(volume)
    volume = np.expand_dims(volume, axis=0)  # Add batch dimension
    return volume

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2, 3))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def get_resized_heatmap(heatmap, shape):
    upscaled_heatmap = np.uint8(255 * heatmap)
    resized_slices = []
    for i in range(upscaled_heatmap.shape[-1]):
        resized_slices.append(cv2.resize(upscaled_heatmap[:, :, i], (shape[1], shape[0])))
    resized_heatmap = np.stack(resized_slices, axis=-1)
    return resized_heatmap

def get_bounding_boxes(heatmap, threshold=0.15, otsu=False):
    p_heatmap = np.copy(heatmap)
    if otsu:
        threshold, p_heatmap = cv2.threshold(
            heatmap, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
    else:
        p_heatmap[p_heatmap < threshold * 255] = 0
        p_heatmap[p_heatmap >= threshold * 255] = 1
    contours = cv2.findContours(p_heatmap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    bboxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        bboxes.append([x, y, x + w, y + h])
    return bboxes

def get_bbox_patches(bboxes, color='r', linewidth=2):
    patches = []
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        patches.append(
            Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                edgecolor=color,
                facecolor='none',
                linewidth=linewidth,
            )
        )
    return patches


rc('animation', html='jshtml')


def create_animation(array, case, heatmap=None, alpha=0.3):
    """Create an animation of a volume"""
    array = np.transpose(array, (2, 0, 1))
    if heatmap is not None:
        heatmap = np.transpose(heatmap, (2, 0, 1))
    fig = plt.figure(figsize=(4, 4))
    images = []
    for idx, image in enumerate(array):
        # plot image without notifying animation
        image_plot = plt.imshow(image, animated=True, cmap='bone')
        aux = [image_plot]
        if heatmap is not None and idx < len(heatmap):
            image_plot2 = plt.imshow(
                heatmap[idx], animated=True, cmap='jet', alpha=alpha, extent=image_plot.get_extent())
            aux.append(image_plot2)

            # add bounding boxes to the heatmap image as animated patches
            bboxes = get_bounding_boxes(heatmap[idx])
            patches = get_bbox_patches(bboxes)
            aux.extend(image_plot2.axes.add_patch(patch) for patch in patches)
        images.append(aux)

    plt.axis('off')
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    plt.title(f'Patient ID: {case}', fontsize=16)
    ani = animation.ArtistAnimation(
        fig, images, interval=5000//len(array), blit=False, repeat_delay=1000)
    plt.close()
    return ani


st.title("3D Image Classification App")

model = tf.keras.models.load_model("https://github.com/mahmoudmagdyhassan/NLP-and-CV/blob/main/New%20folder%20(2)/3d_image_classification%20(3).h5")
new_image_path = "https://github.com/mahmoudmagdyhassan/NLP-and-CV/blob/main/New%20folder%20(2)/study_0942.nii.gz"

number_input_value = st.sidebar.number_input(
    "Enter a number",
    value=0,  # Default value
    min_value=0,  # Minimum allowed value
    max_value=40,  # Maximum allowed value
    step=1,  # Step size
)

new_image = preprocess_image(new_image_path)
input_volume = new_image[0]

# Make prediction
prediction = model.predict(np.expand_dims(input_volume, axis=0))[0]
scores = [1 - prediction[0], prediction[0]]
class_names = ['normal', 'abnormal']

# Display prediction results
for score, name in zip(scores, class_names):
    st.write(
        f'This model is {(100 * score):.2f} percent confident that CT scan is {name}.'
    )



# Display the input volume
st.image(np.squeeze(input_volume[:, :, number_input_value]), caption="CT Scan", use_column_width=True, channels="GRAY")

volume_size = input_volume.shape
last_conv_layer_name = 'conv3d_3'


# Remove last layer's activation
model.layers[-1].activation = None

# Print what the top predicted class is
img_array = np.expand_dims(input_volume, axis=0)

preds = model.predict(img_array)
st.write('Predicted:', preds[0])
heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
# Display Grad-CAM heatmap
fig, ax = plt.subplots()
ax.matshow(np.squeeze(heatmap[:, :, number_input_value]))
st.pyplot(fig)



# Resize heatmap to match the input volume's shape
resized_heatmap = get_resized_heatmap(heatmap, input_volume.shape)

# Display input volume and resized heatmap
fig, ax = plt.subplots(1, 2, figsize=(40, 40))
slice_index = 30  # You can adjust this index as needed

ax[0].imshow(np.squeeze(input_volume[:, :, number_input_value]), cmap='bone')
ax[1].imshow(np.squeeze(input_volume[:, :, number_input_value]), cmap='bone')
ax[1].imshow(np.squeeze(resized_heatmap[:, :, number_input_value]),
                    cmap='jet', alpha=0.3)
st.pyplot(fig)



import cv2
    # show the bounding boxes on the original image
fig, ax = plt.subplots(1, 2, figsize=(40, 40))

ax[0].imshow(np.squeeze(input_volume[:, :, number_input_value]), cmap='bone')
ax[1].imshow(np.squeeze(input_volume[:, :, number_input_value]), cmap='bone')
ax[1].imshow(np.squeeze(resized_heatmap[:, :, number_input_value]),
                    cmap='jet', alpha=0.3)

bboxes = get_bounding_boxes(np.squeeze(resized_heatmap[:, :, number_input_value]))
patches = get_bbox_patches(bboxes)

for patch in patches:
    ax[1].add_patch(patch)

st.pyplot(fig)



