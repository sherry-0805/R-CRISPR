import os
from tensorflow.keras import optimizers
from tensorflow.keras.models import model_from_json
from sklearn import metrics
import dataset_train_test_split
import R_CRISPR
from sklearn import model_selection
import pandas as pd
import tensorflow



os.environ["CUDA_VISIBLE_DEVICES"] = "7"
config = tensorflow.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.7
with tensorflow.Session(config=config) as sess:
    config = tensorflow.ConfigProto()
    config.gpu_options.allow_growth = True


def train_R_CRISPR(x, y, i, data_type = "C"):
    model_dir = "./train_models"
    if os.path.isdir(model_dir):
        pass
    else:
        os.mkdir(model_dir)
    rcrispr = R_CRISPR.R_CRISPR_model()
    optimizer = optimizers.Adam(lr=0.0001)
    rcrispr.compile(loss="binary_crossentropy", optimizer=optimizer)
    rcrispr.fit(x, y, batch_size=10000, epochs=100, shuffle=True)
    rcrispr_j = rcrispr.to_json()

    # save model files
    rcrispr.save_weights(model_dir + "/rcrispr_train_"+"cv_{}".format(i)+".h5")
    with open(model_dir + "/rcrispr_train_"+"cv_{}".format(i)+".json", "w") as f:
        f.write(rcrispr_j)
    print("Finished training!")


def test_R_CRISPR(x,y,i,rcrispr_weights):
    print("Load model json")
    rcrispr_json = "./train_models/rcrispr_train_"+"cv_{}".format(i)+".json"
    rcrispr_json = open(rcrispr_json, 'r')
    rcrispr_model_json = rcrispr_json.read()
    rcrispr_json.close()
    rcrispr_model = model_from_json(rcrispr_model_json)
    print("Load model weights")
    rcrispr_model.load_weights(rcrispr_weights)
    x = x
    y = y
    y_pred = rcrispr_model.predict(x).flatten()

    # auroc
    fpr, tpr, _ = metrics.roc_curve(y, y_pred)
    auroc = metrics.roc_auc_score(y, y_pred)

    # auprc
    prc, rec, _ = metrics.precision_recall_curve(y, y_pred)
    prc[[rec == 0]] = 1.0
    auprc = metrics.auc(rec, prc)
    print("cv_run: ",i," AUROC is ", auroc, "AUPRC is ", auprc)

    data = {'label': y_test, 'pred': y_pred}
    data = pd.DataFrame(data)
    data.to_csv('./cv' + '{}'.format(i) + '_label_pred.csv')
    return 0



kf = model_selection.StratifiedKFold(n_splits=5)
X, y = dataset_train_test_split.load_training_data_C()
splits = kf.split(X,y)
k_sets = []
for train, test in splits:
    X_train = X[train]
    X_test = X[test]
    y_train = y[train]
    y_test = y[test]
    k_sets.append((X_train, X_test, y_train, y_test))

for i in range(len(k_sets)):
    X_train, X_test, y_train, y_test = k_sets[i]
    train_R_CRISPR(X_train, y_train,i, data_type="C")
    rcrispr_weights = "./train_models/rcrispr_train_"+"cv_{}".format(i)+".h5"
    test_R_CRISPR(X_test, y_test, i,rcrispr_weights)











