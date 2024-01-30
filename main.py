import numpy as np
import cv2
import tkinter as tk
from PIL import Image, ImageTk

prototxt_path = 'models/MobileNetSSD_deploy.prototxt'
model_path = 'models/MobileNetSSD_deploy.caffemodel'

min_confidence = 0.2

classes = ['background', 'aeroplane', 'bycicle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
           'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

np.random.seed(543210)
colors = np.random.uniform(0, 255, size=(len(classes), 3))

net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

cap = cv2.VideoCapture(0)

# Create a Tkinter window
root = tk.Tk()
root.title("Object Detection GUI")

# Create a label to display the video feed
label = tk.Label(root)
label.pack(padx=10, pady=10)

orientation_label = tk.Label(root, text="Bottle Orientation: ")
orientation_label.pack(pady=10)

def update_frame():
    _, image = cap.read()
    image_for_bottle = image.copy()
    height, width = image.shape[0], image.shape[1]

    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007, (300, 300), 130)
    net.setInput(blob)
    detected_objects = net.forward()

    for i in range(detected_objects.shape[2]):
        confidence = detected_objects[0][0][i][2]

        if confidence > min_confidence:
            class_index = int(detected_objects[0, 0, i, 1])
            upper_left_x = int(detected_objects[0, 0, i, 3] * width)
            upper_left_y = int(detected_objects[0, 0, i, 4] * height)
            lower_right_x = int(detected_objects[0, 0, i, 5] * width)
            lower_right_y = int(detected_objects[0, 0, i, 6] * height)

            prediction_text = f"{classes[class_index]}: {confidence:.2f}%"

            cv2.rectangle(image, (upper_left_x, upper_left_y), (lower_right_x, lower_right_y), colors[class_index], 3)
            cv2.putText(image, prediction_text,
                        (upper_left_x, upper_left_y - 15 if upper_left_y > 30 else upper_left_y + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[class_index], 2)

            if class_index == 5:
                bottle_roi = image_for_bottle[upper_left_y:lower_right_y, upper_left_x:lower_right_x]

                if not bottle_roi.size == 0:
                    bottle_roi_gray = cv2.cvtColor(bottle_roi, cv2.COLOR_BGR2GRAY)

                    edges = cv2.Canny(bottle_roi_gray, 50, 150)
                    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    contour_image = np.zeros_like(bottle_roi)
                    cv2.drawContours(contour_image, contours, -1, (255, 255, 255), 2)

                    if contours:
                        x, y, w, h = cv2.boundingRect(bottle_roi_gray)
                        aspect_ratio = w / float(h)

                        contour_visualization = np.copy(contour_image)
                        if aspect_ratio > 1:
                            cv2.drawContours(contour_visualization, contours, -1, (255, 255, 255), 2)

                            midpoint_x = contour_image.shape[1] // 2
                            midpoint_y = contour_image.shape[0] // 2

                            closest_column = None
                            closest_distance = float('inf')

                            for x in range(contour_image.shape[1]):
                                y = next((y for y in range(contour_image.shape[0]) if contour_image[y, x].any()), None)

                                if y is not None:
                                    distance = abs(y - midpoint_y)

                                    if distance < closest_distance:
                                        closest_distance = distance
                                        closest_column = x

                            if closest_column is not None:
                                cv2.line(contour_visualization, (closest_column, 0),
                                         (closest_column, contour_image.shape[0] - 1), (0, 255, 0), 2)

                                if closest_column > midpoint_x:
                                    orientation_label.config(text="Bottle is looking right")
                                else:
                                    orientation_label.config(text="Bottle is looking left")
                        else:
                            cv2.drawContours(contour_visualization, contours, -1, (255, 255, 255), 2)

                            midpoint_x = contour_image.shape[1] // 2
                            midpoint_y = contour_image.shape[0] // 2

                            closest_row = None
                            closest_distance = float('inf')

                            for y in range(contour_image.shape[0]):
                                x = next((x for x in range(contour_image.shape[1]) if contour_image[y, x].any()), None)
                                if x is not None:
                                    distance = abs(x - midpoint_x)

                                    if distance < closest_distance:
                                        closest_distance = distance
                                        closest_row = y

                            if closest_row is not None:
                                cv2.line(contour_visualization, (0, closest_row),
                                         (contour_image.shape[1] - 1, closest_row), (0, 255, 0), 2)

                                if closest_row > midpoint_y:
                                    orientation_label.config(text="Bottle is looking down")
                                else:
                                    orientation_label.config(text="Bottle is looking up")

                        # Merge contour_visualization with the original image
                        alpha = 0.5
                        image[upper_left_y:lower_right_y, upper_left_x:lower_right_x] = cv2.addWeighted(
                            image[upper_left_y:lower_right_y, upper_left_x:lower_right_x], alpha,
                                contour_visualization, 1 - alpha, 0)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image_tk = ImageTk.PhotoImage(Image.fromarray(image_rgb))

    label.configure(image=image_tk)
    label.image = image_tk

    root.after(10, update_frame)

update_frame()

root.mainloop()

cap.release()
