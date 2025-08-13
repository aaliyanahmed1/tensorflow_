"""
tf_data.py
Efficient loading, preprocessing, and handling of datasets.
"""
import tensorflow as tf

# Example: Create a tf.data pipeline for image data
def create_data_pipeline():
    # Dummy data: 100 random images (28x28)
    images = tf.random.uniform([100, 28, 28], maxval=255, dtype=tf.float32)
    labels = tf.random.uniform([100], maxval=10, dtype=tf.int32)
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    # Shuffle, batch, and normalize
    dataset = dataset.shuffle(buffer_size=100).batch(16).map(
        lambda x, y: (x / 255.0, y)
    )
    for batch_images, batch_labels in dataset.take(1):
        print(f"Batch images shape: {batch_images.shape}")
        print(f"Batch labels: {batch_labels}")

if __name__ == "__main__":
    create_data_pipeline()
