import os
import cv2
import matplotlib.pyplot as plt

def display_data(root_dir):
    train_dir = os.path.join(root_dir, "bounding_box_train")
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Error: Couldnt find in root directory.")
    
    image_files = [f for f in os.listdir(train_dir) if f.lower().endswith(".jpg")]
    print(f"Found images in training dataset.")

    # display first 5 images, one at a time
    for fname in image_files[:5]:
        full_path = os.path.join(train_dir, fname)
        print(f"Loading...")

        img_bgr = cv2.imread(full_path)
        if img_bgr is None:
            print(f"Could not read image")
            continue

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        plt.imshow(img_rgb)
        plt.title(fname)
        plt.axis("off")
        plt.show()

if __name__ == "__main__":
    root_dir = "Market-1501-v15.09.15"
    display_data(root_dir)