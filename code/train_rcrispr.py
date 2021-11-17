import os
from tensorflow.keras import optimizers

import dataset_train_test_split
import R_CRISPR


def train_R_CRISPR(data_type = "C"):
    model_dir = "./train_models"
    if os.path.isdir(model_dir):
        pass
    else:
        os.mkdir(model_dir)

    # load training data
    if data_type == "A":
        matrix, label = dataset_train_test_split.load_training_data_A()
    elif data_type == "B":
        matrix, label = dataset_train_test_split.load_training_data_B()
    elif data_type == "C":
        matrix, label = dataset_train_test_split.load_training_data_C()
    elif data_type == "D":
        matrix, label = dataset_train_test_split.load_training_data_D()
    elif data_type == "E":
        matrix, label = dataset_train_test_split.load_training_data_E()
    elif data_type == "F":
        matrix, label = dataset_train_test_split.load_training_data_F()
    elif data_type == "G":
        matrix, label = dataset_train_test_split.load_training_data_G()
    elif data_type == "PDH":
        matrix, label = dataset_train_test_split.load_training_data_PDH()

    x = matrix
    y = label
    rcrispr = R_CRISPR.R_CRISPR_model()
    optimizer = optimizers.Adam(lr=0.0001)
    rcrispr.compile(loss="binary_crossentropy", optimizer=optimizer)
    rcrispr.fit(x, y, batch_size=10000, epochs=100, shuffle=True)
    rcrispr_j = rcrispr.to_json()

    # save model files
    rcrispr.save_weights(model_dir + "/rcrispr_train_on_dataB.h5")
    with open(model_dir + "/rcrispr_train_on_dataB.json", "w") as f:
        f.write(rcrispr_j)
    print("Finished training!")


train_R_CRISPR("B")



