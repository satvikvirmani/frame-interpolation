import tensorflow as tf
import os
import time

IMG_HEIGHT = 256
IMG_WIDTH = 448
BATCH_SIZE = 8
EPOCHS = 20
LEARNING_RATE = 1e-4
DATA_DIR = '../vimeo_triplet' # Path to the dataset
TRAIN_LIST = os.path.join(DATA_DIR, 'tri_trainlist.txt')
TEST_LIST = os.path.join(DATA_DIR, 'tri_testlist.txt')

def load_and_preprocess_image(path):
    """Loads and preprocesses a single image."""
    image = tf.io.read_file(path)
    # The Vimeo-90K dataset uses PNG format
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    # The images are already 256x448, but resizing is good practice
    image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])
    return image

def load_triplet(path_suffix):
    """
    Loads the triplet of frames (frame1, frame3 as input; frame2 as target)
    based on the path suffix from the list file (e.g., '00001/0001').
    """
    # The path suffix is a byte string, decode it to a Python string
    path_suffix_str = path_suffix.numpy().decode('utf-8')
    
    # Construct full paths to the three images
    base_path = os.path.join(DATA_DIR, 'sequences', path_suffix_str)
    path1 = os.path.join(base_path, 'im1.png')
    path2 = os.path.join(base_path, 'im2.png')
    path3 = os.path.join(base_path, 'im3.png')

    frame1 = load_and_preprocess_image(path1)
    frame2_target = load_and_preprocess_image(path2) # The middle frame is the target
    frame3 = load_and_preprocess_image(path3)

    # Stack the first and third frames along the channel dimension for the model input
    input_frames = tf.concat([frame1, frame3], axis=-1)
    
    return input_frames, frame2_target

def create_dataset(list_file_path):
    """Creates a tf.data.Dataset from a list file."""
    with open(list_file_path, 'r') as f:
        path_suffixes = [line.strip() for line in f.readlines()]

    dataset = tf.data.Dataset.from_tensor_slices(path_suffixes)
    
    # Use tf.py_function to wrap the Python-based path joining logic
    dataset = dataset.map(lambda x: tf.py_function(load_triplet, [x], [tf.float32, tf.float32]),
                          num_parallel_calls=tf.data.AUTOTUNE)

    # Configure the dataset for performance
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return dataset

# Create the training dataset
train_dataset = create_dataset(TRAIN_LIST)
print(f"✅ Training dataset created successfully with {len(train_dataset)} batches.")