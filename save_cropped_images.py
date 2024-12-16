import os 

def save_cropped_images(cropped_images,folder_path):
    if not os.path.exists('data'):
        os.makedirs('data')
    for key,image in cropped_images.items():
        img_path = os.path.join(folder_path ,key+'.png')
        image.save(img_path)