import numpy as np
import matplotlib.pyplot as plt


def single_image(image, label):
    """Displays a single image"""
    # Ensure the image is a 2D array if it's a grayscale image
    image = image.squeeze()

    # Convert the tensor to a NumPy array if needed
    if not isinstance(image, np.ndarray):
        image = image.numpy()

    # Display the image
    plt.imshow(image, cmap='gray')
    plt.title(f"Label: {label}")
    plt.axis(False)
    plt.show()