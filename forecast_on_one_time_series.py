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
import plotly.express as px
import plotly.graph_objects as go
# %matplotlib inline

# %% [markdown]
# # Data preparation

# %% [markdown]
# ## Reading data

# %%
df_raw = pd.read_csv("train_2.csv")

# %%
df_raw.head()

# %%
df_raw.shape

# %% [markdown]
# ## Cleaning data

# %% [markdown]
# We will use data of only one page

# %%
df_no_na = df_raw.dropna()
df_no_na.head()

# %%
one_page = df_no_na["Page"].sample(1).values[0]
df_one_page = df_no_na[df_no_na["Page"] == one_page].drop("Page", axis=1)
df_one_page.head()

# %% [markdown]
# ## Windows

# %%
window_size = 15
cols = df_one_page.columns
window_data = []
for start_index in range(len(cols) - window_size + 1):
    window_data.append(df_one_page[cols[start_index:start_index + window_size]].values[0])

# %%
len(window_data)

# %% [markdown]
# ## Split between train and test set 

# %%
test_start_index = int(0.85 * len(window_data))
train_data = np.array(window_data[:test_start_index])
test_data = np.array(window_data[test_start_index:])

# %%
x_train = train_data[:, :-1].reshape(len(train_data), window_size - 1, 1)
y_train = train_data[:, -1]
x_test = test_data[:, :-1].reshape(len(test_data), window_size - 1, 1)
y_test = test_data[:, -1]

# %%
x_train.shape, y_train.shape, x_test.shape, y_test.shape


# %% [markdown]
# # Apply LSTM

# %% [markdown]
# ## LSTM without global scaling

# %%
def get_model(x_train):
    input_layer = tf.keras.layers.Input((x_train.shape[1], x_train.shape[2]))
    lstm_layer = tf.keras.layers.LSTM(128, return_sequences=True)(input_layer)
    second_lstm_layer = tf.keras.layers.LSTM(32)(lstm_layer)
    dense_layer = tf.keras.layers.Dense(8)(second_lstm_layer)
    output_layer = tf.keras.layers.Dense(1)(dense_layer)
    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss="mae")
    return model


# %%
model = get_model(x_train)
early_stopping = tf.keras.callbacks.EarlyStopping(patience=100,
                                                 restore_best_weights=True)
model_history = model.fit(x_train, y_train, validation_split=0.1,
                         batch_size=128, epochs=1000,
                        callbacks=early_stopping)


# %%
def visualize_loss(history, title):
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = list(range(len(loss)))
    fig = go.Figure(data=[go.Scatter(x=epochs, y=loss, name="Training loss"),
                   go.Scatter(x=epochs, y=val_loss, name="Validation loss")])
    fig.update_layout(title=title,
                       xaxis_title="Epoch",
                       yaxis_title="Loss")
    fig.show()


# %%
visualize_loss(model_history, 
               "Training of a LSTM model without scaling");

# %%
preds_test = model.predict(x_test)
mean_absolute_error(preds_test[:, 0], y_test)

# %% [markdown]
# ## LSTM with scaling

# %% [markdown]
# ### Scale features

# %%
mean_clicks = np.mean(train_data)
mean_clicks

# %%
x_train_scaled = x_train / mean_clicks
y_train_scaled = y_train / mean_clicks
x_test_scaled =  x_test / mean_clicks
y_test_scaled = y_test / mean_clicks

# %% [markdown]
# ### Train models

# %%
model_scaled = get_model(x_train)
early_stopping = tf.keras.callbacks.EarlyStopping(patience=100,
                                                 restore_best_weights=True)

scaled_model_history = model_scaled.fit(x_train_scaled, y_train_scaled, 
                                        validation_split=0.1,
                                        batch_size=128, epochs=1000,
                                       callbacks=early_stopping)

# %%
visualize_loss(scaled_model_history, 
               "Training of a LSTM model with scaling");

# %%
preds_scaled_test = model_scaled.predict(x_test_scaled)
mean_absolute_error(preds_scaled_test[:, 0] * mean_clicks, y_test)

# %% [markdown]
# # Visualize results

# %% [markdown]
# ## Visualize global predictions vs reality

# %%
preds_without_scaling_vs_real = pd.DataFrame({"predictions": preds_test[:, 0],
                                             "reality": y_test})
preds_with_scaling_vs_real = pd.DataFrame({"predictions": preds_scaled_test[:, 0] * mean_clicks,
                                             "reality": y_test})

preds_vs_real = pd.concat([preds_without_scaling_vs_real.assign(model="model_without_scaling"),
                          preds_with_scaling_vs_real.assign(model="model_with_scaling")])
preds_vs_real.head()

# %%
px.scatter(preds_vs_real, x="predictions", y="reality", color="model",
          trendline="ols")

# %% [markdown]
# ## Visualize predictions vs reality as time series

# %%
days = df_one_page.columns[-len(preds_test):]

fig = go.Figure(data=[go.Line(x=days, 
                              y=y_test, 
                              name="Number of clicks"),
                      go.Line(x=days, 
                              y=preds_test[:, 0], 
                              name="Predictions with model not using scaling"),
                     go.Line(x=days, 
                              y=preds_scaled_test[:, 0] * mean_clicks, 
                              name="Predictions with model using global scaling")])
fig.update_layout(title="Predictions vs reality on test set",
                   xaxis_title="Day",
                   yaxis_title="Clicks")

# %%
