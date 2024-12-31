from google.cloud import vision
import os
from PIL import Image
import io

class TextDetector:
    def __init__(self):
        """Initializes the TextDetector with a Vision API client."""
        self.client = vision.ImageAnnotatorClient()

    def detect_text(self, image_path):
        """Detects text in an image.

        Args:
            image_path (str): Path to the image file.

        Returns:
            str: Detected text from the image.

        Raises:
            Exception: If the Vision API returns an error.
        """
        with open(image_path, "rb") as image_file:
            content = image_file.read()
            image = vision.Image(content=content)

        image_context = vision.ImageContext(language_hints=["ar"])

        # Perform text detection
        response = self.client.text_detection(image=image, image_context=image_context)
        texts = response.text_annotations

        if response.error.message:
            raise Exception(f"Error: {response.error.message}")

        # Return the detected text (only the first annotation contains the full text block)
        if texts:
            return texts[0].description
        return ""
    
    def detect_texts(self, folder_path):
        """Detects text in all images in a folder.
        Args:
            folder_path (str): Path to the folder containing image files.
        Returns:
            dict: A dictionary mapping image filenames to their detected text.
        """

        detected_texts = {}
        for filename in os.listdir(folder_path):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(folder_path, filename)
                try:
                    detected_text = self.detect_text(image_path)
                    detected_texts[filename] = detected_text
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
        return detected_texts

if __name__ == "__main__":
    image_path = "ID_F.png"  # Replace with your image file path
    detector = TextDetector()
    try:
        detected_text = detector.detect_text(image_path)
        print(f"Detected text: {detected_text}")
    except Exception as e:
        print(e)
