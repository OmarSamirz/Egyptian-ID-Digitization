from data_labeller import DataLabeller
import cv2
from save_cropped_images import save_cropped_images
from detect_text import TextDetector
from split_image import ImageSegmenter

def crop_and_transcribe(img_path):
    data_labeller = DataLabeller()
    segments = data_labeller.predict_image(img_path)
    segmenter = ImageSegmenter(data_labeller)
    cropped_images = segmenter.segment_and_crop(img_path)
    save_cropped_images(cropped_images,'./data')
    text_detector = TextDetector()
    transcribed_texts = text_detector.detect_texts('./data')
    return transcribed_texts


def main():
    # data_labeller = DataLabeller()
    # imgs_path = 'test_image.jpg'
    # true_imgs_output_path = ''
    # false_imgs_output_path = ''
    # # data_labeller.imgs_labelling(imgs_path, true_imgs_output_path, false_imgs_output_path)
    # print(data_labeller.predict_image(imgs_path))
    # img = data_labeller.img_labelling(imgs_path)[0]
    # cv2.imshow('Image', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    texts = crop_and_transcribe('test_image.jpg')
    print(texts)
    with open('texts.txt', 'w') as f:
        for key, value in texts.items():
            value =  value.replace("\n", " ")
            f.write(f'{key}: {value}\n')


if __name__ == '__main__':
    main()