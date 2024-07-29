import cv2
import numpy as np
import pymongo
from bson import Binary
import io
from PIL import Image
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model,Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.optimizers import Adam



# Connect to MongoDB
client = pymongo.MongoClient('mongodb://localhost:27017/')
db = client['deepfake_detection']
collection = db['images']

def preprocess_and_save_to_mongodb(video_path, label):
    """Extract frames from a video and save to MongoDB."""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error opening video file {video_path}")
        return
    
    i = 1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize and normalize image
        frame = cv2.resize(frame, (150, 150))  # Resize to match model input
        frame = frame / 255.0  # Normalize to [0, 1]
        
        # Convert image to bytes
        _, buffer = cv2.imencode('.jpg', (frame * 255).astype(np.uint8))
        image_bytes = Binary(buffer.tobytes())
        
        # Save to MongoDB
        collection.insert_one({'image': image_bytes, 'label': label})
        i += 1
    
    cap.release()


preprocess_and_save_to_mongodb('fake_video.mp4', 0)  # 0 for fake
preprocess_and_save_to_mongodb('real_video.mp4', 1)  # 1 for real

def load_data_from_mongodb(batch_size=32, img_height=150, img_width=150):
    """Load images and labels from MongoDB."""
    cursor = collection.find()
    images = []
    labels = []
    
    for doc in cursor:
        image_bytes = doc['image']
        image = Image.open(io.BytesIO(image_bytes))
        image = image.resize((img_height, img_width))
        image = np.array(image) / 255.0  # Normalize to [0, 1]
        
        images.append(image)
        labels.append(doc['label'])
    
    images = np.array(images)
    labels = np.array(labels)
    
    return images, labels

# Load data
images, labels = load_data_from_mongodb()

# Split data into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

# Create TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)
val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(32)



# Build the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_dataset,
    epochs=10,
    validation_data=val_dataset
)

# Save the model
model.save('deep_fake_detection_model.h5')



# Load the model
model = load_model('deep_fake_detection_model.h5')

def predict_video(video_path):
    """Predict if the video is real or fake."""
    # Extract frames
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (150, 150))
        frame = frame / 255.0
        frames.append(frame)
    
    cap.release()
    
    # Make predictions
    frames = np.array(frames)
    predictions = model.predict(frames)
    average_prediction = np.mean(predictions)
    
    return 'Fake' if average_prediction > 0.5 else 'Real'

# Example usage
result = predict_video('path_to_test_video.mp4')
print(f'The video is: {result}')




