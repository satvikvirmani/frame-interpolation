import tensorflow as tf

class MultiFrameInterpolator:
    def __init__(self, model, input_size=(128, 224)):
        self.model = model
        self.input_size = input_size  # must match model training

    def predict_between(self, f_start, f_end, num_intermediate):
        H, W = f_start.shape[:2]
        in_h, in_w = self.input_size

        # Normalize to [0,1] for model input (in case f_start/f_end are uint8)
        f1_resized = tf.image.resize(tf.cast(f_start, tf.float32) / 255.0, (in_h, in_w))
        f2_resized = tf.image.resize(tf.cast(f_end, tf.float32) / 255.0, (in_h, in_w))

        predicted_frames = []

        for i in range(1, num_intermediate + 1):
            model_input = tf.expand_dims(tf.concat([f1_resized, f2_resized], axis=-1), axis=0)
            pred_small = self.model(model_input, training=False)[0]

            # pred_small is [0,1], resize back to full size
            pred_full = tf.image.resize(pred_small, (H, W))

            # Convert to uint8 RGB [0,255]
            pred_uint8 = tf.clip_by_value(pred_full * 255.0, 0, 255)
            predicted_frames.append((i, tf.cast(pred_uint8, tf.uint8)))

        return predicted_frames

    def display_results(self, f_start, f_end, predicted_frames):
        """
        Optional: visualize start, end, and predicted frames
        """
        import matplotlib.pyplot as plt

        total_frames = [f_start] + [f for _, f in predicted_frames] + [f_end]
        plt.figure(figsize=(20, 4))
        for i, frame in enumerate(total_frames):
            plt.subplot(1, len(total_frames), i+1)
            plt.imshow(frame.numpy().astype("uint8"))
            plt.axis("off")
            plt.title(f"Frame {i+1}")
        plt.show()