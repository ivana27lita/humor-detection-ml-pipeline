import os
import tensorflow as tf
import tensorflow_transform as tft
from tensorflow.keras import layers
from tfx.components.trainer.fn_args_utils import FnArgs

LABEL_KEY = 'humor'
FEATURE_KEY = 'text'


def transformed_name(key):
    """Renaming transformed features"""
    return key + "_xf"


def gzip_reader_fn(filenames):
    """Loads compressed data"""
    return tf.data.TFRecordDataset(filenames, compression_type='GZIP')


def input_fn(file_pattern, tf_transform_output, num_epochs, batch_size=64) -> tf.data.Dataset:
    """Get post_transform feature & create batches of data"""

    # Get post_transform feature spec
    transform_feature_spec = (
        tf_transform_output.transformed_feature_spec().copy()
    )

    # Create batches of data
    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transform_feature_spec,
        reader=gzip_reader_fn,
        num_epochs=num_epochs,
        label_key=transformed_name(LABEL_KEY)
    )

    return dataset


# Vocabulary size and number of words in a sequence
VOCAB_SIZE = 10000
SEQUENCE_LENGTH = 100

vectorize_layer = layers.TextVectorization(
    standardize='lower_and_strip_punctuation',
    max_tokens=VOCAB_SIZE,
    output_mode='int',
    output_sequence_length=SEQUENCE_LENGTH
)


def model_builder(hp):
    """Build machine learning model"""
    inputs = tf.keras.Input(
        shape=(1,), name=transformed_name(FEATURE_KEY), dtype=tf.string)

    reshaped_narrative = tf.reshape(inputs, [-1])
    x = vectorize_layer(reshaped_narrative)
    x = layers.Embedding(VOCAB_SIZE, hp.get(
        'embedding_dim', 8), name='embedding')(x)
    x = layers.Bidirectional(layers.LSTM(hp.get('lstm_units', 16)))(x)
    for _ in range(hp.get('num_layers', 1)):
        x = layers.Dense(hp.get('dense_units', 32), activation='relu')(x)
    x = layers.Dropout(hp.get('dropout_rate', 0.5))(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(hp.get('learning_rate', 0.001)),
        metrics=[tf.keras.metrics.BinaryAccuracy()]
    )

    model.summary()
    return model


def _get_serve_tf_examples_fn(model, tf_transform_output):
    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples):
        feature_spec = tf_transform_output.raw_feature_spec()
        feature_spec.pop(LABEL_KEY)

        parsed_features = tf.io.parse_example(
            serialized_tf_examples, feature_spec)
        transformed_features = model.tft_layer(parsed_features)

        # get predictions using the transformed features
        return model(transformed_features)

    return serve_tf_examples_fn


def run_fn(fn_args: FnArgs) -> None:
    log_dir = os.path.join(os.path.dirname(fn_args.serving_model_dir), 'logs')
    hp = fn_args.hyperparameters['values']

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, update_freq='batch'
    )

    early_stop_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_binary_accuracy',
        mode='max',
        verbose=1,
        patience=10
    )

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        fn_args.serving_model_dir,
        monitor='val_binary_accuracy',
        mode='max',
        verbose=1,
        save_best_only=True
    )

    callbacks = [
        tensorboard_callback,
        early_stop_callback,
        model_checkpoint_callback
    ]

    # Load the transform output
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

    # Create batches of data
    train_set = input_fn(fn_args.train_files,
                         tf_transform_output, num_epochs=hp.get('epochs', 10))
    val_set = input_fn(fn_args.eval_files,  tf_transform_output,
                       num_epochs=hp.get('epochs', 10))

    vectorize_layer.adapt(
        [j[0].numpy()[0] for j in [
            i[0][transformed_name(FEATURE_KEY)]
            for i in list(train_set)
        ]]
    )

    # Build the model
    model = model_builder(hp)

    # Train the model
    model.fit(
        x=train_set,
        validation_data=val_set,
        callbacks=callbacks,
        steps_per_epoch=fn_args.train_steps,
        validation_steps=fn_args.eval_steps,
        epochs=hp.get('epochs', 7)
    )

    signatures = {
        'serving_default': _get_serve_tf_examples_fn(
            model, tf_transform_output
        ).get_concrete_function(
            tf.TensorSpec(
                shape=[None],
                dtype=tf.string,
                name='examples'
            )
        )
    }

    model.save(
        fn_args.serving_model_dir,
        save_format='tf',
        signatures=signatures
    )
