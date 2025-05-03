import os
import cv2
import random
from tqdm import tqdm

# --- CONFIG ---
input_dir = '/test/images'
label_dir = '/test/labels'
output_dir = 'test'  # Unified output folder

# Paths for images and labels
output_image_dir = os.path.join(output_dir, 'images')
output_label_dir = os.path.join(output_dir, 'labels')

# Cropping aspect ratios and resizing dimensions
aspect_ratios = [(4, 3), (16, 9), (1, 1)]  # Cropping aspect ratios
resize_dims = [(640, 480), (800, 600)]    # Resizing (W, H)

# Percentages
crop_percentage = 0.8  # 80% cropping, 20% resizing

# Ensure the output folders exist
os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_label_dir, exist_ok=True)

# --- UTILS ---
def load_labels(label_path):
    if not os.path.exists(label_path):
        return []
    with open(label_path, 'r') as f:
        return [list(map(float, line.strip().split())) for line in f]

def save_labels(label_path, labels):
    with open(label_path, 'w') as f:
        for label in labels:
            f.write(" ".join(map(str, label)) + "\n")

def adjust_bbox_crop(bbox, crop_x, crop_y, crop_w, crop_h, orig_w, orig_h):
    cls, x, y, w, h = bbox
    abs_x = x * orig_w
    abs_y = y * orig_h
    abs_w = w * orig_w
    abs_h = h * orig_h

    x1 = abs_x - abs_w / 2
    y1 = abs_y - abs_h / 2
    x2 = abs_x + abs_w / 2
    y2 = abs_y + abs_h / 2

    # Clipping the bounding box to the crop
    x1_new = max(x1, crop_x)
    y1_new = max(y1, crop_y)
    x2_new = min(x2, crop_x + crop_w)
    y2_new = min(y2, crop_y + crop_h)

    if x2_new <= x1_new or y2_new <= y1_new:
        return None  # Bounding box is fully outside

    new_abs_w = x2_new - x1_new
    new_abs_h = y2_new - y1_new
    new_abs_x = (x1_new + x2_new) / 2 - crop_x
    new_abs_y = (y1_new + y2_new) / 2 - crop_y

    return [
        int(cls),
        new_abs_x / crop_w,
        new_abs_y / crop_h,
        new_abs_w / crop_w,
        new_abs_h / crop_h
    ]

# --- MAIN PROCESSING ---
image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
random.shuffle(image_files)

num_crop = int(len(image_files) * crop_percentage)
crop_files = image_files[:num_crop]
resize_files = image_files[num_crop:]

for img_file in tqdm(image_files, desc="Processing images"):
    img_path = os.path.join(input_dir, img_file)
    label_path = os.path.join(label_dir, os.path.splitext(img_file)[0] + '.txt')

    img = cv2.imread(img_path)
    if img is None:
        continue
    h, w = img.shape[:2]
    labels = load_labels(label_path)

    base_name = os.path.splitext(img_file)[0]

    if img_file in crop_files:
        # Cropping the image
        ar_w, ar_h = random.choice(aspect_ratios)
        target_ratio = ar_w / ar_h

        if w / h > target_ratio:
            crop_w = int(h * target_ratio)
            crop_h = h
        else:
            crop_w = w
            crop_h = int(w / target_ratio)

        crop_x = (w - crop_w) // 2
        crop_y = (h - crop_h) // 2

        cropped_img = img[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w]
        new_labels = []

        for bbox in labels:
            new_bbox = adjust_bbox_crop(bbox, crop_x, crop_y, crop_w, crop_h, w, h)
            if new_bbox:
                new_labels.append(new_bbox)

        out_img_path = os.path.join(output_image_dir, f"{base_name}.jpg")
        out_label_path = os.path.join(output_label_dir, f"{base_name}.txt")
        cv2.imwrite(out_img_path, cropped_img)
        save_labels(out_label_path, new_labels)

    else:
        # Resizing the image
        new_w, new_h = random.choice(resize_dims)
        resized_img = cv2.resize(img, (new_w, new_h))

        # YOLO format remains normalized, so no changes to bounding boxes
        out_img_path = os.path.join(output_image_dir, f"{base_name}.jpg")
        out_label_path = os.path.join(output_label_dir, f"{base_name}.txt")
        cv2.imwrite(out_img_path, resized_img)
        save_labels(out_label_path, labels)
