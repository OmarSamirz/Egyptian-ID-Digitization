import cv2
import numpy as np

# Read your image (replace 'path_to_image.jpg' with your actual image path)
image = cv2.imread('./datasets/CDC_OCR.v3i.yolov8-obb/train/images/1-2-_jpg.rf.72d8c647fe7d240c18924558d7db08fc.jpg')

# Ensure the image is loaded
if image is None:
    raise FileNotFoundError("Image not found. Please check the path.")

# Get image dimensions
img_height, img_width = image.shape[:2]

# Labels data
labels = [
    (7, 0.90234375, 0.28254847645429376, 0.576171875, 0.2825484764542937, 0.576171875, 0.4501385041551248, 0.90234375, 0.4501385041551248),
    (0, 0.9052734375, 0.4612188365650971, 0.53125, 0.4612188365650971, 0.53125, 0.6246537396121885, 0.9052734375, 0.6246537396121885),
    (6, 0.921875, 0.700831024930748, 0.4472656250000001, 0.7008310249307479, 0.4472656250000001, 0.8033240997229919, 0.921875, 0.803324099722992),
    (1, 0.3710937500000001, 0.6800554016620498, 0.12304687500000006, 0.6800554016620498, 0.12304687500000006, 0.7936288088642659, 0.3710937500000001, 0.7936288088642659),
    (2, 0.3574218750000001, 0.8296398891966759, 0.13378906250000006, 0.8296398891966759, 0.13378906250000006, 0.9127423822714681, 0.3574218750000001, 0.9127423822714681),
]

for label in labels:
    class_id, x1, y1, x2, y2, x3, y3, x4, y4 = label
    
    # Denormalize coordinates
    points = np.array([
        [x1 * img_width, y1 * img_height],
        [x2 * img_width, y2 * img_height],
        [x3 * img_width, y3 * img_height],
        [x4 * img_width, y4 * img_height]
    ], dtype=np.int32)
    
    # Draw the quadrilateral
    cv2.polylines(image, [points], isClosed=True, color=(0, 255, 0), thickness=2)

# Display the image
cv2.imshow("Image with Labels", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
