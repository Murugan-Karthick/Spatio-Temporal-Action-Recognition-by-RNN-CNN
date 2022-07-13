# Spatio-Temporal-Action-Recognition with a CNN-RNN Architecture

**Training a action recognizer with transfer learning and a recurrent model on the UCF101 dataset.**

## Demo

![alt text](https://github.com/Murugan-Karthick/Spatio-Temporal-Action-Recognition-by-RNN-CNN/blob/main/animation.gif)

  ShavingBeard: 42.38%<br>
  TennisSwing: 16.67%<br>
  PlayingCello: 15.84%<br>
  CricketShot: 13.08%<br>
  Punch: 12.03%

We will be using the [UCF101 dataset](https://www.crcv.ucf.edu/data/UCF101.php)
to build our action recognizer. The dataset consists of videos categorized into different actions, like

1. cricket shot, 
2. punching, 
3. biking, etc. 

A video consists of an ordered sequence of frames. Each frame contains *spatial*
information, and the sequence of those frames contains *temporal* information. To model
both of these aspects, we use a hybrid architecture that consists of convolutions
(for spatial processing) as well as recurrent layers (for temporal processing).
Specifically, we'll use a Convolutional Neural Network (CNN) and a Recurrent Neural
Network (RNN) consisting of [GRU layers].

## Data collection

In order to make training time to low, we will be using a
subsampled version of the original UCF101 dataset. download the dataset from [UCF101 dataset](https://git.io/JGc31) link.

## Requirements
Before run the code you should run below the lines for installing dependencies
```bash
  pip install tensorflow
  pip install -q git+https://github.com/tensorflow/docs
  pip install imutils
  pip install opencv-python
  pip install matplotlib
```

## Data preprocessing
One of the many challenges of training action recognizer is figuring out a way to feed
the videos to a network. [This blog post](https://blog.coast.ai/five-video-classification-methods-implemented-in-keras-and-tensorflow-99cad29cc0b5)
discusses five such methods. Since a video is an ordered sequence of frames, we could
just extract the frames and put them in a 3D tensor. But the number of frames may differ
from video to video which would prevent us from stacking them into batches
(unless we use padding). As an alternative, we can **save video frames at a fixed
interval until a maximum frame count is reached**. In this example we will do
the following:

1. Capture the frames of a video.

```
def load_video(path, max_frames=0, resize=(IMG_SIZE, IMG_SIZE)):
    cap = cv2.VideoCapture(path)
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = crop_center_square(frame)
            frame = cv2.resize(frame, resize)
            frame = frame[:, :, [2, 1, 0]]
            frames.append(frame)

            if len(frames) == max_frames:
                break
    finally:
        cap.release()
    return np.array(frames)
```
2. Extract frames from the videos until a maximum frame count is reached.

```
# Extract features from the frames of the current video.
for i, batch in enumerate(frames):
    video_length = batch.shape[0]
    length = min(MAX_SEQ_LENGTH, video_length)
    for j in range(length):
        temp_frame_features[i, j, :] = feature_extractor.predict(
            batch[None, j, :]
        )
    temp_frame_mask[i, :length] = 1  # 1 = not masked, 0 = masked

frame_features[idx,] = temp_frame_features.squeeze()
frame_masks[idx,] = temp_frame_mask.squeeze()
```
3. In the case, where a video's frame count is lesser than the maximum frame count we
will pad the video with zeros.

## Feature Extractor
We can use a pre-trained network to extract meaningful features from the extracted frames. The [`Keras Applications`](https://keras.io/api/applications/) module provides a number of state-of-the-art models pre-trained on the [ImageNet-1k dataset](http://image-net.org/). We will be using the [InceptionV3 model](https://arxiv.org/abs/1512.00567) for this purpose.

```
def build_feature_extractor():
    feature_extractor = keras.applications.InceptionV3(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    preprocess_input = keras.applications.inception_v3.preprocess_input

    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3))
    preprocessed = preprocess_input(inputs)

    outputs = feature_extractor(preprocessed)
    return keras.Model(inputs, outputs, name="feature_extractor")


feature_extractor = build_feature_extractor()
```

The labels of the videos are strings. Neural networks do not understand string values, so they must be converted to some numerical form before they are fed to the model. Here we will use the [`StringLookup`](https://keras.io/api/layers/preprocessing_layers/categorical/string_lookup) layer encode the class labels as integers.

```
label_processor = keras.layers.StringLookup(
    num_oov_indices=0, vocabulary=np.unique(train_df["tag"])
)
print(label_processor.get_vocabulary())
Output: ['CricketShot', 'PlayingCello', 'Punch', 'ShavingBeard', 'TennisSwing']
```

## The sequence model

Now, we can feed this data to a sequence model consisting of recurrent layers like `GRU`.

```
# Utility for our sequence model.
def get_sequence_model():
    class_vocab = label_processor.get_vocabulary()

    frame_features_input = keras.Input((MAX_SEQ_LENGTH, NUM_FEATURES))
    mask_input = keras.Input((MAX_SEQ_LENGTH,), dtype="bool")

    # Refer to the following tutorial to understand the significance of using `mask`:
    # https://keras.io/api/layers/recurrent_layers/gru/
    x = keras.layers.GRU(16, return_sequences=True)(
        frame_features_input, mask=mask_input
    )
    x = keras.layers.GRU(8)(x)
    x = keras.layers.Dropout(0.4)(x)
    x = keras.layers.Dense(8, activation="relu")(x)
    output = keras.layers.Dense(len(class_vocab), activation="softmax")(x)

    rnn_model = keras.Model([frame_features_input, mask_input], output)

    rnn_model.compile(
        loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    return rnn_model
```

## Training
```
def run_experiment():
    filepath = "./tmp/action_recognizer"
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath, save_weights_only=True, save_best_only=True, verbose=1
    )

    seq_model = get_sequence_model()
    history = seq_model.fit(
        [train_data[0], train_data[1]],
        train_labels,
        validation_split=0.3,
        epochs=EPOCHS,
        callbacks=[checkpoint],
    )

    seq_model.load_weights(filepath)
    _, accuracy = seq_model.evaluate([test_data[0], test_data[1]], test_labels)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")

    return history, seq_model

# for training a sequential model
train_performance, sequence_model = run_experiment()
```

## Model Performance
![alt text](https://github.com/Murugan-Karthick/Spatio-Temporal-Action-Recognition-by-RNN-CNN/blob/main/results.png)

## Inference

For inference we need to do video preprocessing before input to the model

```
def prepare_single_video(frames):
    frames = frames[None, ...]
    frame_mask = np.zeros(shape=(1, MAX_SEQ_LENGTH,), dtype="bool")
    frame_features = np.zeros(shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32")

    for i, batch in enumerate(frames):
        video_length = batch.shape[0]
        length = min(MAX_SEQ_LENGTH, video_length)
        for j in range(length):
            frame_features[i, j, :] = feature_extractor.predict(batch[None, j, :])
        frame_mask[i, :length] = 1  # 1 = not masked, 0 = masked

    return frame_features, frame_mask
```

## Making Prediction
```
def sequence_prediction(path):
    class_vocab = label_processor.get_vocabulary()

    frames = load_video(os.path.join("test", path))
    frame_features, frame_mask = prepare_single_video(frames)
    probabilities = sequence_model.predict([frame_features, frame_mask])[0]

    for i in np.argsort(probabilities)[::-1]:
        print(f"  {class_vocab[i]}: {probabilities[i] * 100:5.2f}%")
    return frames
    
test_video = np.random.choice(test_df["video_name"].values.tolist())
print(f"Test video path: {test_video}")
test_frames = sequence_prediction(test_video)
to_gif(test_frames[:MAX_SEQ_LENGTH])
```
