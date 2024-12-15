from PIL import Image
import matplotlib.pyplot as plt
from data_labeller import DataLabeller


class ImageSegmenter:
    def __init__(self, data_labeller):
        self.data_labeller = data_labeller

    def segment_and_crop(self, img_path):
        """Segments and crops parts of an image based on predictions from the data labeller.

        Args:
            img_path (str): Path to the image file.

        Returns:
            dict: A dictionary with segment class names as keys and cropped images as values.
        """

        segments = self.data_labeller.predict_image(img_path)
        image = Image.open(img_path)

        cropped_images = {}
        for segment in segments:
            x = int(segment["x"])
            y = int(segment["y"])
            width = int(segment["width"])
            height = int(segment["height"])

            cropped = image.crop((x - width / 2, y - height / 2, x + width / 2, y + height / 2))
            cropped_images[segment["class"]] = cropped

        return cropped_images

    def display_segments(self, cropped_images):
        """Displays cropped segments with their labels.

        Args:
            cropped_images (dict): Dictionary of cropped images with class names as keys.
        """
        fig, axes = plt.subplots(1, len(cropped_images), figsize=(15, 5))

        for ax, (name, img) in zip(axes, cropped_images.items()):
            ax.imshow(img)
            ax.set_title(name)
            ax.axis("off")

        plt.tight_layout()
        plt.show()

# Example Usage:
data_labeller = DataLabeller()  
segmenter = ImageSegmenter(data_labeller)
img_path = 'test_image.jpg'
cropped_images = segmenter.segment_and_crop(img_path)
segmenter.display_segments(cropped_images)
