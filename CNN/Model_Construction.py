import numpy as np
import tensorflow as tf
import os
import datetime
import re
import matplotlib.pyplot as plt

from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from sklearn.model_selection import GroupKFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

from collections import Counter



# Training settings

IMAGE_SIZE = 96

image_dir = r"C:\Users\joshy\OneDrive\Documents\Kravitz Lab\freezeData\CNN Data\\"

# Datetime ranges for a given quality (0 to 1 is "good" = 0, 2 to 3 is "bad" = 1)

# cutoffs = [
#     ["0309", [
#         datetime.datetime(2026, 3, 7, 12, 0, 0), datetime.datetime(2026, 3, 8, 2, 0, 0),
#         datetime.datetime(2026, 3, 8, 6, 10, 0), datetime.datetime(2026, 3, 10, 5, 59, 59)
#     ]], ["0318", [
#         datetime.datetime(2026, 3, 17, 17, 17, 0), datetime.datetime(2026, 3, 18, 4, 30, 0),
#         datetime.datetime(2026, 3, 18, 12, 0, 0), datetime.datetime(2026, 3, 18, 13, 0, 0)
#     ]], ["0322_0214", [
#         datetime.datetime(2026, 3, 19, 10, 52, 0), datetime.datetime(2026, 3, 27, 14, 7, 0),
#         datetime.datetime(2026, 2, 14, 16, 40, 0), datetime.datetime(2026, 2, 15, 12, 0, 0)
#     ]], ["0506", [
#         datetime.datetime(2026, 5, 6, 10, 55, 0), datetime.datetime(2026, 5, 6, 16, 40, 0),
#         datetime.datetime(2026, 5, 6, 19, 20, 0), datetime.datetime(2026, 5, 6, 21, 59, 59)
#     ]]
# ]

cutoffs = [
    ["0309", [
        datetime.datetime(2026, 3, 7, 12, 0, 0), datetime.datetime(2026, 3, 8, 2, 0, 0),
        datetime.datetime(2026, 3, 9, 5, 40, 0), datetime.datetime(2026, 3, 10, 5, 59, 59)
    ]], ["0318", [
        datetime.datetime(2026, 3, 17, 17, 17, 0), datetime.datetime(2026, 3, 18, 0, 0, 0),
        datetime.datetime(2026, 3, 18, 12, 0, 0), datetime.datetime(2026, 3, 18, 13, 0, 0)
    ]], ["0322_0214", [
        datetime.datetime(2026, 3, 19, 10, 52, 0), datetime.datetime(2026, 3, 27, 14, 7, 0),
        datetime.datetime(2026, 2, 14, 16, 40, 0), datetime.datetime(2026, 2, 15, 12, 0, 0)
    ]], ["0506", [
        datetime.datetime(2026, 5, 6, 10, 55, 0), datetime.datetime(2026, 5, 6, 16, 40, 0),
        datetime.datetime(2026, 5, 6, 19, 20, 0), datetime.datetime(2026, 5, 6, 21, 59, 59)
    ]]
]



# Image processing functions

def getTime(filename):
    # Take the file name
    filename = os.path.basename(filename)

    # Search for underscores - we're undecided on how to format the datetime in an image's filename yet
    match = re.match(r"(\d{8}_?\d{6})", filename)
    if match:
        time_str = match.group(1).replace('_', '')
        time = datetime.datetime.strptime(time_str, '%Y%m%d%H%M%S')
    else:
        raise ValueError(f"No valid timestamp found in filename: {filename}")
    
    return time

def load_image(filename, colors=3):

    img = tf.io.read_file(filename)

    img = tf.image.decode_png(
        img,
        channels=colors
    )

    timestamp = getTime(filename)
    quality = -1
    freeze_id = -1
    for i in range(len(cutoffs)):
        cutoff = cutoffs[i][1]
        if (cutoff[0] is not None) and (cutoff[0] <= timestamp) and (timestamp <= cutoff[1]):
            quality = 0
            freeze_id = cutoffs[i][0]
            break
        elif (cutoff[2] is not None) and (cutoff[2] <= timestamp) and (timestamp <= cutoff[3]):
            quality = 1
            freeze_id = cutoffs[i][0]
            break
    

    return img, quality, freeze_id, timestamp

augment = tf.keras.Sequential([
    tf.keras.layers.RandomFlip(
        "horizontal_and_vertical"
    ),
    tf.keras.layers.RandomRotation(
        factor=1.0,
        fill_mode="constant"
    ),
])

def preprocess(image, label, training=False):

    image = tf.image.convert_image_dtype(
        image,
        tf.float32
    )

    image = tf.image.resize(
        image,
        (IMAGE_SIZE,IMAGE_SIZE)
    )

    if training:
        image = augment(image, training=True)


    return image, label

def repeat_augment(image, label):

    ds = tf.data.Dataset.from_tensors((image, label))

    ds = ds.repeat(4)

    ds = ds.map(
        lambda x, y: preprocess(x, y, training=True),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    return ds

if __name__ == "__main__":
    # Loading in initial list of images

    filelist = [
        os.path.join(image_dir + date[0] + r'\\', f)
        for date in cutoffs
        for f in os.listdir(image_dir + date[0] + r'\\')
    ]

    samples = []
    labels = []
    freeze_ids = []

    with ThreadPoolExecutor(max_workers=10) as executor:

        futures = [
            executor.submit(load_image, filename)
            for filename in filelist
        ]

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Loading images"
        ):

            img, quality, freeze_id, _ = future.result()
            if quality != -1:
                samples.append(img)
                labels.append(quality)
                freeze_ids.append(freeze_id)
    # nums =[0,0,0,0]
    # for file in filelist:
    #     img, quality, freeze_id = load_image(file)

    counts = Counter(freeze_ids)

    print("Images per freeze:")
    for freeze, count in sorted(counts.items()):
        print(f"{freeze}: {count}")
    
    X = np.stack(samples)
    y = np.array(labels)

    # print(X.shape)

    # Specifying model architecture

    def build_model():

        model = tf.keras.Sequential([

            tf.keras.layers.Conv2D(
                16,
                3,
                padding="same",
                activation="relu",
                input_shape=(IMAGE_SIZE,IMAGE_SIZE,3),
            ),

            tf.keras.layers.MaxPooling2D(),

            tf.keras.layers.Conv2D(
                32,
                3,
                padding="same",
                activation="relu",
            ),

            tf.keras.layers.MaxPooling2D(),

            tf.keras.layers.Conv2D(
                64,
                3,
                padding="same",
                activation="relu",
            ),

            tf.keras.layers.GlobalAveragePooling2D(),

            tf.keras.layers.Dense(
                32,
                activation="relu"
            ),

            tf.keras.layers.Dropout(
                0.5
            ),

            tf.keras.layers.Dense(
                1,
                activation="sigmoid"
            )
        ])


        # Model Training

        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=1e-3,
                # weight_decay=1e-4
            ),
            loss="binary_crossentropy",
            metrics=[
                tf.keras.metrics.AUC(name="auc"),
            ],
        )
        return model
    
    # Setting up cross-validation

    cv = GroupKFold(n_splits=len(cutoffs))

    aucs = []
    cms = []

    cutoffs_trans = [list(row) for row in zip(*cutoffs)]
    freeze_names = cutoffs_trans[1]
    freeze_ids = np.array(freeze_ids)

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y, groups=freeze_ids)):
        
        val_freezes = np.unique(freeze_ids[val_idx])
        val_name = val_freezes[0]
        best_model_name = f"best_model_fold{fold}_{val_name}.keras"
        final_model_name = f"freezeCNN_fold{fold}_{val_name}.keras"

        print(f"Validation freeze(s): {val_name}")

        # Defining the training and validation datasets and doing data augmentation

        X_train = X[train_idx]
        X_val   = X[val_idx]

        y_train = y[train_idx]
        y_val   = y[val_idx]
        

        train_dataset = tf.data.Dataset.from_tensor_slices(
            (X_train, y_train)
        )

        val_dataset = tf.data.Dataset.from_tensor_slices(
            (X_val, y_val)
        )


        train_dataset = (
            train_dataset
            .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
            .cache()
            .flat_map(repeat_augment)
            .shuffle(4 * len(X_train))
            .batch(32)
            .prefetch(tf.data.AUTOTUNE)
        )

        val_dataset = (
            val_dataset
            .map(preprocess)
            .batch(32)
            .prefetch(tf.data.AUTOTUNE)
        )

        # Building the model and creating callbacks

        tf.keras.backend.clear_session()

        model = build_model()

        callbacks = [

            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=10,
                restore_best_weights=True
            ),

            tf.keras.callbacks.ModelCheckpoint(
                best_model_name,
                monitor="val_loss",
                save_best_only=True
            ),

            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=5,
                verbose=1
            )

        ]

        # Training the model

        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=100,
            callbacks=callbacks,
        )

        history_dict = history.history

        epochs = range(1, len(history_dict["loss"]) + 1)

        plt.figure(figsize=(10,4))

        # Loss
        plt.subplot(1,2,1)
        plt.plot(epochs, history_dict["loss"], label="Training")
        plt.plot(epochs, history_dict["val_loss"], label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Binary Cross-Entropy Loss")
        plt.title(f"Fold {fold}: {val_name}")
        plt.grid(True)
        plt.legend()

        # AUC
        plt.subplot(1,2,2)
        plt.plot(epochs, history_dict["auc"], label="Training")
        plt.plot(epochs, history_dict["val_auc"], label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("AUC")
        plt.title(f"Fold {fold}: {val_name}")
        plt.grid(True)
        plt.legend()

        plt.tight_layout()

        plt.savefig(f"learning_curve_fold{fold}_{val_name}.png", dpi=300)
        plt.close()

        loss, auc = model.evaluate(val_dataset)

        y_prob = model.predict(val_dataset).flatten()

        y_pred = (y_prob >= 0.5).astype(int)

        y_true = np.concatenate(
            [labels.numpy() for _, labels in val_dataset],
            axis=0
        )
        cm = confusion_matrix(y_true, y_pred)


        aucs.append(auc)
        cms.append(cm)
        model.save(final_model_name)
    

    for cm in cms:
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["Good", "Bad"]
        )

        disp.plot(cmap="Blues")

        plt.title(f"Fold {fold}: {val_name}")
        plt.show()

    print(cms)

    print("\nCross-validation AUCs:", aucs)
    print("Mean:", np.mean(aucs))
    print("Std:", np.std(aucs))