from data_labeller import DataLabeller
import cv2

def main():
    data_labeller = DataLabeller()
    imgs_path = 'test_image.jpg'
    true_imgs_output_path = ''
    false_imgs_output_path = ''
    # data_labeller.imgs_labelling(imgs_path, true_imgs_output_path, false_imgs_output_path)
    print(data_labeller.predict_image(imgs_path))
    img = data_labeller.img_labelling(imgs_path)[0]
    cv2.imshow('Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



if __name__ == '__main__':
    main()