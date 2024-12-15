import cv2
from inference_sdk import InferenceHTTPClient

import os
api_key =  os.getenv('API_KEY')
class DataLabeller:
    def __init__(self, 
                 api_url: str = 'https://detect.roboflow.com', ##
                 api_key: str = api_key,##
                 model_id: str = 'cdc_ocr/1') -> None:
        self.model_id = model_id
        self.CLIENT = InferenceHTTPClient(api_url=api_url, api_key=api_key)

    def predict_image(self, img_path: str) -> dict:
        result = self.CLIENT.infer(img_path, model_id=self.model_id)
        return result['predictions']


    def _save_img(self, img: cv2.typing.MatLike, output_path: str) -> None:
        cv2.imwrite(output_path, img)


    def img_labelling(self, img_path: str) -> tuple[cv2.typing.MatLike, bool]:
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
        os.makedirs(true_imgs_output_path, exist_ok=True)
        os.makedirs(false_imgs_output_path, exist_ok=True)
        images = os.listdir(imgs_path)

        for img in images:
            img_path = os.path.join(imgs_path, img)
            cv2_img, is_true_img = self.img_labelling(img_path)
            img_path = os.path.join(true_imgs_output_path if is_true_img == True else false_imgs_output_path, img)
            self._save_img(cv2_img, img_path)
