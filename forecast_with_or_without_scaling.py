# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.7.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
# %matplotlib inline

# %% [markdown]
# # Data preparation

# %% [markdown]
# ## Reading data

# %%
train_1 = pd.read_csv("train_1.csv")

# %%
train_1.head()

# %%
train_1.shape

# %% [markdown]
# ## Cleaning data

# %% [markdown]
# We will only keep instantly usable data, as we do not need much for experiment

# %%
train_cleaned = train_1.dropna().drop("Page", axis=1)
train_cleaned.head()

# %%
train_cleaned.shape

# %% [markdown]
# ## Split between train and test set 

# %%
df_train, df_test = train_test_split(train_cleaned, test_size=0.2)

# %% [markdown]
# We will only try to predict what happened on the 29th of July 2015, given data about the previous four weeks

# %%
features = list(pd.date_range(start="2015-07-01",
                             end="2015-07-28").strftime("%Y-%m-%d"))
target = "2015-07-29"

# %%
x_train = df_train[features]
y_train = df_train[target]
x_test = df_test[features]
y_test = df_test[target]

# %%
x_train_reshaped = x_train.values.reshape((len(x_train), len(x_train.columns), 1))
x_test_reshaped = x_test.values.reshape((len(x_test), len(x_train.columns), 1))


# %% [markdown]
# # Compare LSTM modelizations

# %% [markdown]
# ## LSTM without scaling

# %%
def get_model(x_train_reshaped):
    input_layer = tf.keras.layers.Input((x_train_reshaped.shape[1], x_train_reshaped.shape[2]))
    lstm_layer = tf.keras.layers.LSTM(32)(input_layer)
    dense_layer = tf.keras.layers.Dense(1)(lstm_layer)
    model = tf.keras.Model(inputs=input_layer, outputs=dense_layer)
    model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss="mae")
    return model


# %%
model = get_model(x_train_reshaped)
early_stopping = tf.keras.callbacks.EarlyStopping(patience=20,
                                                 restore_best_weights=True)
no_scaling_model_history = model.fit(x_train_reshaped, y_train, validation_split=0.1,
                                     batch_size=128, epochs=500,
                                    callbacks=early_stopping)


# %%
def visualize_loss(history, title):
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(len(loss))
    plt.figure()
    plt.plot(epochs, loss, "b", label="Training loss")
    plt.plot(epochs, val_loss, "r", label="Validation loss")
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


# %%
visualize_loss(no_scaling_model_history, 
               "Training of a LSTM model without scaling");

# %%
preds_test = model.predict(x_test_reshaped)
mean_absolute_error(preds_test[:, 0], y_test)

# %% [markdown]
# ## LSTM with global (features) scaling

# %% [markdown]
# ### Sclale data

# %%
train_mean = df_train.mean().mean()
train_mean

# %%
x_train_scaled = x_train_reshaped / train_mean
y_train_scaled = y_train / train_mean
x_test_scaled = x_test_reshaped / train_mean
y_test_scaled = y_test / train_mean

# %% [markdown]
# ### Train model

# %%
model_global_scaling = get_model(x_train_scaled)
early_stopping = tf.keras.callbacks.EarlyStopping(patience=20,
                                                 restore_best_weights=True)
global_scaling_model_history = model.fit(x_train_scaled, y_train_scaled, validation_split=0.1,
                                     batch_size=128, epochs=500,
                                    callbacks=early_stopping)

# %%
visualize_loss(global_scaling_model_history, 
               "Training of a LSTM model with global scaling");

# %%
preds_global_scaling_test = model.predict(x_test_scaled)
mean_absolute_error(preds_global_scaling_test[:, 0] * train_mean, y_test)

# %% [markdown]
# ## LSTM with time series scaling

# %% [markdown]
# ### Scale data

# %%
df_train_features_target = df_train[features + [target]]
df_test_features_target = df_test[features + [target]]

# %%
df_train_means = df_train_features_target.mean(axis=1).replace(0, 1)
df_test_means = df_test_features_target.mean(axis=1).replace(0, 1)

# %%
df_train_scaled = df_train_features_target.assign(**{
    f: df_train_features_target[f] / df_train_means for f in df_train_features_target.columns
})
df_test_scaled = df_test_features_target.assign(**{
    f: df_test_features_target[f] / df_test_means for f in df_test_features_target.columns
})

# %%
df_train_scaled.head()

# %%
x_train_scaled = df_train_scaled[features]
y_train_scaled = df_train_scaled[target]
x_test_scaled = df_test_scaled[features]
y_test_scaled = df_test_scaled[target]

# %%
x_train_scaled_reshaped = x_train_scaled.values.reshape((len(x_train), len(x_train.columns), 1))
x_test_scaled_reshaped = x_test_scaled.values.reshape((len(x_test), len(x_train.columns), 1))

# %% [markdown]
# ### Train models

# %%
model_scaled = get_model(x_train_reshaped)
early_stopping = tf.keras.callbacks.EarlyStopping(patience=20,
                                                 restore_best_weights=True)

scaled_model_history = model_scaled.fit(x_train_scaled_reshaped, y_train_scaled, 
                                        validation_split=0.1,
                                        batch_size=128, epochs=500,
                                       callbacks=early_stopping)

# %%
visualize_loss(scaled_model_history, 
               "Training of a LSTM model with scaling");

# %%
preds_scaled_test = model_scaled.predict(x_test_scaled_reshaped)
mean_absolute_error(preds_scaled_test[:, 0] * df_test_means, y_test)

# %% [markdown]
# ## Model with scaling and sample weights

# %%
model_scaling_weights = get_model(x_train_reshaped)
early_stopping = tf.keras.callbacks.EarlyStopping(patience=20,
                                                 restore_best_weights=True)

scaled_model_with_weights_history = model_scaling_weights.fit(x_train_scaled_reshaped,
                                                    y_train_scaled, 
                                                    validation_split=0.1,
                                                     batch_size=128, epochs=500, 
                                                    sample_weight=df_train_means,
                                                   callbacks=early_stopping)

# %%
visualize_loss(scaled_model_with_weights_history, 
               "Training of a LSTM model with scaling and sample weights");

# %%
preds_scaled_weighted_test = model_scaling_weights.predict(x_test_scaled_reshaped)
mean_absolute_error(preds_scaled_weighted_test[:, 0] * df_test_means,
                    y_test)


# %% [markdown]
# # Comparison with other models

# %% [markdown]
# ## Dense model

# %%
def get_ff_model(x_train):
    input_layer = tf.keras.layers.Input((x_train.shape[1]))
    dense_1 = tf.keras.layers.Dense(256, activation="relu")(input_layer)
    dense_2 = tf.keras.layers.Dense(64, activation="relu")(dense_1)
    dense_3 = tf.keras.layers.Dense(16, activation="relu")(dense_2)
    dense_4 = tf.keras.layers.Dense(4, activation="relu")(dense_3)
    output_layer = tf.keras.layers.Dense(1, activation="relu")(dense_3)
    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss="mae")
    return model


# %%
ff_model = get_ff_model(x_train)

# %%
early_stopping = tf.keras.callbacks.EarlyStopping(patience=20,
                                                 restore_best_weights=True)

ff_model_history = ff_model.fit(x_train, y_train, validation_split=0.1,
                                  batch_size=128, epochs=1000,
                                callbacks=early_stopping)

# %%
visualize_loss(ff_model_history, 
               "Training of a feedforward model");

# %%
preds_ff_test = ff_model.predict(x_test)
mean_absolute_error(preds_ff_test[:, 0], y_test)

# %% [markdown]
# ## Random forest without feature engineering

# %%
rf = RandomForestRegressor(criterion="mae", n_jobs=10, max_depth=4,
                          n_estimators=20)
rf.fit(x_train, y_train)

# %%
preds_rf_test = rf.predict(x_test)
mean_absolute_error(preds_rf_test, y_test)


# %% [markdown]
# ## Attention model

# %% [markdown]
# ### Unscaled

# %%
def get_attention_model(x_train_reshaped):
    input_layer = tf.keras.layers.Input((x_train_reshaped.shape[1], x_train_reshaped.shape[2]))
    query_layer = tf.keras.layers.LSTM(32, return_sequences=True)(input_layer)
    value_layer = tf.keras.layers.LSTM(32, return_sequences=True)(input_layer)
    attention_layer = tf.keras.layers.Attention()([query_layer, value_layer])
    dense_layer = tf.keras.layers.Dense(1)(attention_layer)
    model = tf.keras.Model(inputs=input_layer, outputs=dense_layer)
    model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss="mae")
    return model


# %%
attention_model = get_attention_model(x_train_reshaped)
early_stopping = tf.keras.callbacks.EarlyStopping(patience=20,
                                                 restore_best_weights=True)
attention_model_history = attention_model.fit(x_train_reshaped, y_train, validation_split=0.1,
                                             batch_size=128, epochs=500,
                                             callbacks=early_stopping)

# %%
visualize_loss(attention_model_history, 
               "Training of a LSTM model with attention");

# %%
preds_attention_test = attention_model.predict(x_test_reshaped)
mean_absolute_error(preds_attention_test[:, 0].reshape(-1),
                    y_test)

# %% [markdown]
# ### Scaled

# %%
attention_scaling_model = get_attention_model(x_train_scaled_reshaped)
early_stopping = tf.keras.callbacks.EarlyStopping(patience=20,
                                                 restore_best_weights=True)
attention_scaling_model_history = attention_scaling_model.fit(
    x_train_scaled_reshaped, y_train_scaled, validation_split=0.1,
     batch_size=128, epochs=500,
     callbacks=early_stopping)

# %%
visualize_loss(attention_scaling_model_history, 
               "Training of a LSTM model with attention and scaling");

# %%
preds_attention_scaling_test = attention_scaling_model.predict(x_test_scaled_reshaped)
mean_absolute_error(preds_attention_scaling_test[:, 0].reshape(-1) * df_test_means,
                    y_test)

# %%
