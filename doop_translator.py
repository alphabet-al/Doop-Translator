import tensorflow as tf

class_names = ['A', 'B', 'C', 'D', '.', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(64, 64, 3)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),  # Increase dropout to reduce overfitting
    tf.keras.layers.Dense(27)  # Output layer for 27 classes
])

model.load_weights('doop_model.weights.h5')

predict_path = r'C:\Users\alpha\Projects\Doop Translator\predict4'

prediction_ds = tf.keras.utils.image_dataset_from_directory(
    predict_path,
    labels=None,
    batch_size=32,
    image_size=(64,64),
    shuffle=False    
    )

prediction_ds = prediction_ds.map(lambda x: (x/255.))

predictions = model.predict(prediction_ds)
predicted_labels = tf.argmax(predictions, axis=1)
mapped_predicted_labels = [class_names[i] for i in predicted_labels]

print(mapped_predicted_labels)