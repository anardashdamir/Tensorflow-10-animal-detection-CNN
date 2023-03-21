import tensorflow as tf
import matplotlib.pyplot as plt


def load_and_prep_image(filename, img_shape=224, scale=True):
    """
    Reads in an image from filename, turns it into a tensor and reshapes into
    (224, 224, 3).

    Parameters
    ----------
    filename (str): string filename of target image
    img_shape (int): size to resize target image to, default 224
    scale (bool): whether to scale pixel values to range(0, 1), default True
    """
    # Read in the image
    img = tf.io.read_file(filename)
    # Decode it into a tensor
    img = tf.image.decode_jpeg(img)
    # Resize the image
    img = tf.image.resize(img, [img_shape, img_shape])
    if scale:
        # Rescale the image (get all values between 0 and 1)
        return img / 255.0
    else:
        return img


def pred_and_plot(model, filename, class_names, scale):
    """
    Imports an image located at filename, makes a prediction on it with
    a trained model and plots the image with the predicted class as the title.
    """
    # Import the target image and preprocess it
    img = load_and_prep_image(filename, scale=scale)

    # Make a prediction
    pred = model.predict(tf.expand_dims(img, axis=0))

    # Get the predicted class
    if len(pred[0]) > 1:  # check for multi-class
        pred_class = class_names[pred.argmax()]  # if more than one output, take the max
    else:
        pred_class = class_names[int(tf.round(pred)[0][0])]  # if only one output, round

    # Plot the image and predicted class
    plt.figure(figsize=(8, 5))
    plt.imshow(img / 255)
    plt.title(f"Prediction: {pred_class}")
    plt.axis(False)
    plt.show()


class_names = [
    "butterfly",
    "cat",
    "chicken",
    "cow",
    "dog",
    "elephant",
    "horse",
    "ragno",
    "sheep",
    "squirrel",
]

model = tf.keras.models.load_model("model_1.h5")
pred_and_plot(model, "tofiq_cat.jpg", class_names, False)
