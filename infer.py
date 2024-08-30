import sys
import tensorflow as tf
import numpy as np
from PIL import Image

IMG_SIZE = 224
MODEL_PATH = 'models/best_model.keras'
BATCH_SIZE = 32  # You can adjust the batch size as needed

def preprocess_images(image_paths):
    images = []
    for image_path in image_paths:
        image = tf.io.read_file(image_path)
        image = tf.image.decode_image(image, channels=3)
        image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
        image = tf.expand_dims(image, axis=0)
        images.append(image)
    
    return tf.concat(images, axis=0)

def save_with_orientation(image_path, orientation):
    image = Image.open(image_path)
    rotated_image = image.rotate(-orientation * 90, expand=True)
    rotated_image.save(image_path, "JPEG", quality=95)
    print(f"Rotated and saved {image_path} with orientation {orientation}")

def main(image_paths):
    # Load the trained model
    model = tf.keras.models.load_model(MODEL_PATH)

    # Process images in batches
    for i in range(0, len(image_paths), BATCH_SIZE):
        batch_paths = image_paths[i:i + BATCH_SIZE]
        
        try:
            # Preprocess the images
            input_batch = preprocess_images(batch_paths)

            # Predict the orientation for the batch
            predictions = model.predict(input_batch)
            predicted_labels = np.argmax(predictions, axis=1)  # Get the index of the highest prediction

            # Save each image with the predicted orientation
            for image_path, predicted_label in zip(batch_paths, predicted_labels):
                save_with_orientation(image_path, predicted_label)
                print(f"Processed {image_path}: Predicted orientation label - {predicted_label}")
        except Exception as e:
            print(f"Error processing batch {batch_paths}: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 infer.py image1.jpg image2.jpg ...")
        sys.exit(1)
    
    image_paths = sys.argv[1:]
    main(image_paths)
