from ultralytics import YOLO
import os
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from PIL import Image

# Load pretrained YOLOv8 model and specify the validation data
model = YOLO("3. results/weights/best.pt")

# Run testing on the test data and print the results
metrics = model.val()

# TEST_IMAGE_DIR: Directory containing test images
# TEST_LABEL_DIR: Directory containing corresponding label files
TEST_IMAGE_DIR = "datasets/test/images/"
TEST_LABEL_DIR = "datasets/test/labels/"

# List test image files and create corresponding label file names
test_image_files = os.listdir(TEST_IMAGE_DIR)
test_label_files = [f.replace('.jpg', '.txt') for f in test_image_files]

def count_bounding_boxes(label_file):
    # Count the number of bounding boxes in a label file
    with open(label_file, 'r') as file:
        boxes = sum(1 for _ in file)
    return boxes

true_counts, pred_counts = [], []

# Iterate through the test label files
for label_file in test_label_files:
    true = count_bounding_boxes(os.path.join(TEST_LABEL_DIR, label_file))
    true_counts.append(true)

    # Make predictions using the YOLOv8 model on the corresponding test image
    results = model(os.path.join(TEST_IMAGE_DIR, label_file.replace('.txt', '.jpg')))
    pred = len(results[0].boxes.cls)
    pred_counts.append(pred)

    print(f"True count: {true}")
    print(f"Pred count: {pred}")

# Calculate Mean Absolute Error (MAE), Root Mean Square Error (RMSE), and Mean Absolute Percentage Error (MAPE)
mae = mean_absolute_error(true_counts, pred_counts)
rmse = np.sqrt(mean_squared_error(true_counts, pred_counts))
mape = np.mean(np.abs(np.array(true_counts) - np.array(pred_counts)) / np.array(true_counts))

print(metrics)
#results_dict: {'metrics/precision(B)': 0.9571946209733537, 'metrics/recall(B)': 0.943093826100903, 'metrics/mAP50(B)': 0.9766882427435997, 'metrics/mAP50-95(B)': 0.7472715145623505, 'fitness': 0.7702131873804755}

print(f"MAE on validation set: {mae}")
print(f"RMSE on validation set: {rmse}")
print(f"MAPE on validation set: {mape}%")

"""
results_dict: {'metrics/precision(B)': 0.9571946209733537, 'metrics/recall(B)': 0.943093826100903, 'metrics/mAP50(B)': 0.9766882427435997, 'metrics/mAP50-95(B)': 0.7472715145623505, 'fitness': 0.7702131873804755}
MAE on validation set: 1.6911764705882353
RMSE on validation set: 2.1454466146866924
MAPE on validation set: 0.09885747675198596%
"""

# Get example for two pictures
# Image path
image_path = "datasets/test/images/2022-03-26-10-00-061_jpg.rf.0f5417fc126dac336063bcf6ce75ca41.jpg"
# Perform inference on the image
results = model(image_path)
# Show the results
for r in results:
    im_array = r.plot()  # plot a BGR numpy array of predictions
    im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
    im.show()  # show image
    im.save('output/result_1.jpg')  # save image