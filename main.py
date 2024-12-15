from data_labeller import DataLabeller


def main():
    data_labeller = DataLabeller()
    imgs_path = 'test_image.jpg'
    true_imgs_output_path = ''
    false_imgs_output_path = ''
    # data_labeller.imgs_labelling(imgs_path, true_imgs_output_path, false_imgs_output_path)
    print(data_labeller.predict_image(imgs_path))


if __name__ == '__main__':
    main()