
import kerastuner as kt
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_transform as tft
from tfx.components.tuner.component import TunerFnResult
import os

LABEL_KEY = "humor"
FEATURE_KEY = "text"

VOCAB_SIZE = 10000
SEQUENCE_LENGTH = 100
NUM_EPOCHS = 5

vectorize_layer = layers.TextVectorization(
    standardize="lower_and_strip_punctuation",
    max_tokens=VOCAB_SIZE,
    output_mode='int',
    output_sequence_length=SEQUENCE_LENGTH
)

def transformed_name(key):
    return key + "_xf"

def gzip_reader_fn(filenames):
    return tf.data.TFRecordDataset(filenames, compression_type="GZIP")

def input_fn(file_pattern, tf_transform_output, num_epochs, batch_size=64):
    transform_feature_spec = (
        tf_transform_output.transformed_feature_spec().copy()
    )
    
    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transform_feature_spec,
        reader=gzip_reader_fn,
        num_epochs=num_epochs,
        label_key=transformed_name(LABEL_KEY),
    )
    
    return dataset

def build_model(hp):
    embedding_dim = 16

    hp_units_1 = hp.Int('units_1', min_value=32, max_value=128, step=32)
    hp_units_2 = hp.Int('units_2', min_value=16, max_value=64, step=16)
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3])

    inputs = tf.keras.Input(shape=(1,), name=transformed_name(FEATURE_KEY), dtype=tf.string)
    reshaped_text = tf.reshape(inputs, [-1])
    x = vectorize_layer(reshaped_text)
    x = layers.Embedding(VOCAB_SIZE, embedding_dim, name="embedding")(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(hp_units_1, activation='relu')(x)
    x = layers.Dense(hp_units_2, activation="relu")(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
        metrics=[tf.keras.metrics.BinaryAccuracy()]
    )

    return model

def tuner_fn(fn_args):
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)
    train_dataset = input_fn(fn_args.train_files[0], tf_transform_output, NUM_EPOCHS)
    eval_dataset = input_fn(fn_args.eval_files[0], tf_transform_output, NUM_EPOCHS)
    
    # Adapt the vectorize_layer with the training dataset
    vectorize_layer.adapt(train_dataset.map(lambda x, y: x[transformed_name(FEATURE_KEY)]))
    
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

    tuner = kt.Hyperband(
        build_model,
        objective='val_binary_accuracy',
        max_epochs=10,
        factor=3,
        directory=fn_args.working_dir,
        project_name='kt_hyperband'
    )

    return TunerFnResult(
        tuner=tuner,
        fit_kwargs={
            "callbacks": [early_stopping_callback],
            'x': train_dataset,
            'validation_data': eval_dataset,
            'steps_per_epoch': fn_args.train_steps,
            'validation_steps': fn_args.eval_steps
        }
    )
