import cv2
from inference_sdk import InferenceHTTPClient

import os
api_key =  os.getenv('ROBOFLOW_API_KEY')
class DataLabeller:
    def __init__(self, 
                 api_url: str = 'https://detect.roboflow.com', ##
                 api_key: str = api_key,##
                 model_id: str = 'cdc_ocr/1') -> None:
        self.model_id = model_id
        self.CLIENT = InferenceHTTPClient(api_url=api_url, api_key=api_key)

    def predict_image(self, img_path: str) -> dict:
        """Predict image using the Roboflow Inference API.

        Args:
            img_path (str): Path to the input image.

        Returns:
            dict: A dictionary containing the predictions for the image.
        """
        
        result = self.CLIENT.infer(img_path, model_id=self.model_id)
        return result['predictions']


    def _save_img(self, img: cv2.typing.MatLike, output_path: str) -> None:
        """Save the image to the specified output path.

        Args:
            img (cv2.typing.MatLike): The image to be saved.
            output_path (str): The path to which the image is saved.
        """
        
        cv2.imwrite(output_path, img)


    def img_labelling(self, img_path: str) -> tuple[cv2.typing.MatLike, bool]:
        """Label the image using the predictions from the Roboflow Inference API.

        Args:
            img_path (str): Path to the input image.

        Returns:
            tuple[cv2.typing.MatLike, bool]: A tuple containing the labelled image
                and a boolean indicating whether the image has 5 or 7 predictions.
        """

        predictions = self.predict_image(img_path)

        img = cv2.imread(img_path)
        for pred in predictions:
            x, y, w, h = pred['x'], pred['y'], pred['width'], pred['height']
            left = int(x - w / 2)
            top = int(y - h / 2)
            right = int(x + w / 2)
            bottom = int(y + h / 2)

            cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)

        return (img, True if len(predictions) == 5 or len(predictions) == 7 else False)
    
    def imgs_labelling(self, imgs_path: str, true_imgs_output_path: str, false_imgs_output_path: str) -> None:
        """Label all images in the given directory and save them to the specified output paths.

        Args:
            imgs_path (str): Path to the directory containing the input images.
            true_imgs_output_path (str): Path to the directory where the images with 5 or 7 predictions are saved.
            false_imgs_output_path (str): Path to the directory where the images without 5 or 7 predictions are saved.
        """
        
        os.makedirs(true_imgs_output_path, exist_ok=True)
        os.makedirs(false_imgs_output_path, exist_ok=True)
        images = os.listdir(imgs_path)

        for img in images:
            img_path = os.path.join(imgs_path, img)
            cv2_img, is_true_img = self.img_labelling(img_path)
            img_path = os.path.join(true_imgs_output_path if is_true_img == True else false_imgs_output_path, img)
            self._save_img(cv2_img, img_path)
