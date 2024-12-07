from data_labeller import DataLabeller


def main():
    data_labeller = DataLabeller()
    imgs_path = ''
    true_imgs_output_path = ''
    false_imgs_output_path = ''
    data_labeller.imgs_labelling(imgs_path, true_imgs_output_path, false_imgs_output_path)


if __name__ == '__main__':
    main()