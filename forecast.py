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

# %%
train_1 = pd.read_csv("train_1.csv")

# %%
train_1.head()

# %%
train_1.shape

# %% [markdown]
# # Clean

# %%
train_cleaned = train_1.dropna().drop("Page", axis=1)
train_cleaned.head()

# %%
train_cleaned.shape

# %% [markdown]
# # Model without scaling 

# %%
df_train, df_test = train_test_split(train_cleaned, test_size=0.2)

# %%
features = list(pd.date_range(start="2015-07-01",
                             end="2016-06-30").strftime("%Y-%m-%d"))
target = "2016-07-01"

# %%
x_train = df_train[features]
y_train = df_train[target]
x_test = df_test[features]
y_test = df_test[target]

# %%
x_train_reshaped = x_train.values.reshape((len(x_train), len(x_train.columns), 1))
x_test_reshaped = x_test.values.reshape((len(x_test), len(x_train.columns), 1))


# %%
def get_model(x_train_reshaped):
    input_layer = tf.keras.layers.Input((x_train_reshaped.shape[1], x_train_reshaped.shape[2]))
    lstm_layer = tf.keras.layers.LSTM(1)(input_layer)

    model = tf.keras.Model(inputs=input_layer, outputs=lstm_layer)
    model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss="mae")
    return model


# %%
model = get_model(x_train_reshaped)
model.fit(x_train_reshaped, y_train, validation_split=0.1,
         batch_size=128, epochs=20)

# %%
preds_test = model.predict(x_test_reshaped)

# %%
np.sum(np.mean(np.abs(preds_test[:, 0] - y_test)))

# %% [markdown]
# # Model with scaling

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

# %%
model_scaled = get_model(x_train_reshaped)
history_scaled = model_scaled.fit(x_train_scaled_reshaped, y_train_scaled, validation_split=0.1,
                                 batch_size=128, epochs=20)

# %%
preds_scaled_test = model_scaled.predict(x_test_scaled_reshaped)

# %%
np.sum(np.mean(np.abs(preds_scaled_test[:, 0] * df_test_means - y_test)))

# %% [markdown]
# # Scaling with sample weights

# %%
model_scaling_weights = get_model(x_train_reshaped)
history_scaling_weights = model_scaling_weights.fit(x_train_scaled_reshaped,
                                                    y_train_scaled, 
                                                    validation_split=0.1,
                                                     batch_size=128, epochs=20, 
                                                    sample_weight=df_train_means)

# %%
preds_scaled_weighted_test = model_scaling_weights.predict(x_test_scaled_reshaped)

# %%
np.sum(np.mean(np.abs(preds_scaled_weighted_test[:, 0] * df_test_means - y_test)))

# %%
