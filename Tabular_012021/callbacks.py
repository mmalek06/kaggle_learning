import os

from tensorflow import keras


def get_callbacks(_type: str, model_dir: str, log_dir: str):
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10,
                                                   min_delta=1e-4)
    models_count = len(os.listdir(model_dir)) + 1
    model_checkpoint = keras.callbacks.ModelCheckpoint(
        os.path.join('saved_models', f'{_type}_{models_count}.h5'),
        save_best_only=True)
    curr_log_dir = os.path.join(log_dir, f'{_type}{models_count}')
    tensorboard = keras.callbacks.TensorBoard(curr_log_dir, histogram_freq=1, profile_batch=10)

    return early_stopping, model_checkpoint, tensorboard
