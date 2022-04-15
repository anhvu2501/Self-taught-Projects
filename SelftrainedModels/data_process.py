from pathlib import Path

import numpy as np
from keras.preprocessing import image


def load_data():
    p = Path("./flowers/")

    dirs = p.glob("*")

    image_data = []
    labels = []

    image_paths = []

    labels_dict = {"daisy": 0, "dandelion": 1, "rose": 2, "sunflower": 3, "tulip": 4}

    for folder_dir in dirs:
        label = str(folder_dir).split('/')[-1]
        # print(label)

        cnt = 0
        # print(folder_dir)

        # Iterate over folder_dir and pick all images of flowers

        for img_path in folder_dir.glob("*.jpg"):
            # print(img_path)
            img = image.load_img(img_path, target_size=(64, 64))
            img_array = image.img_to_array(img)
            image_data.append(img_array)
            labels.append(labels_dict[label])
            cnt += 1
        # print(image_data[0].shape)
        # print(cnt)

    # print(len(image_data))
    # print(len(labels))

    X = np.array(image_data)
    Y = np.array(labels)
    return X, Y

# Draw some flowers
# def drawImg(img, label):
#     plt.imshow(img)
#     plt.show()
# drawImg(X[0] / 255.0, label[0])
