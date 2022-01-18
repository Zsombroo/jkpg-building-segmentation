import tensorflow as tf

from image_slicing_helper import extract_file_names_from_input_data
from tf_ds_gen import load_training_data
from u_net_model import create_model

from constants import ENSEMBLE_SIZE
from constants import EPOCHS
from constants import IMAGE_SHAPE
from constants import MODEL_NAME
from constants import MODEL_SAVE_FOLDER
from constants import X_PATH


extract_file_names_from_input_data(X_PATH)

performances = []
for i in range(ENSEMBLE_SIZE):
    print(f'{MODEL_NAME}_{i} / {ENSEMBLE_SIZE}')
    train, val, test = load_training_data()
    model = create_model(f'{MODEL_NAME}_{i}')
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            mode='min',
            verbose=1,
            restore_best_weights=True,
            patience=20,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            f'{MODEL_SAVE_FOLDER}/{MODEL_NAME}_{i}.h5',
            monitor='val_loss',
            mode='min',
            verbose=1,
            save_best_only=True,
        )
    ]
    history = model.fit(
        train,
        epochs=EPOCHS,
        validation_data=val,
        callbacks=callbacks
    )
    performance = model.evaluate(test)
    performances.append(performance)

# TODO: Print test results to give an initial feedback on the models performance

models = [tf.keras.models.load_model(f'{MODEL_SAVE_FOLDER}/{MODEL_NAME}_{i}.h5') for i in range(ENSEMBLE_SIZE)]
model_input = tf.keras.Input(shape=IMAGE_SHAPE)
model_outputs = [model(model_input) for model in models]
ensemble_output = tf.keras.layers.Average()(model_outputs)
ensemble_model = tf.keras.Model(inputs=model_input, outputs=ensemble_output, name=MODEL_NAME)
ensemble_model.save(f'{MODEL_SAVE_FOLDER}/{MODEL_NAME}.h5')
