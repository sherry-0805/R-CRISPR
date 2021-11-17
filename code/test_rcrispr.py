import matplotlib.pyplot as plt
from tensorflow.keras.models import model_from_json
from sklearn import metrics

import dataset_train_test_split



def test_R_CRISPR(rcrispr_json, rcrispr_weights, test_data = "GUIDE_II", fig_auroc = False, fig_auprc = False):

    if test_data == "GUIDE_II":
        matrix_test, label_test = dataset_train_test_split.load_testing_data_GUIDE_II()
        print("Load test data GUIDE_II")
    elif test_data == "GUIDE_III":
        matrix_test, label_test = dataset_train_test_split.load_testing_data_GUIDE_III()
        print("Load test data GUIDE_III")
    else:
        matrix_test, label_test = dataset_train_test_split.load_training_data_C()
        print("Load test data CIRCLE")

    print("Load model json")
    rcrispr_json = open(rcrispr_json, 'r')
    rcrispr_model_json = rcrispr_json.read()
    rcrispr_json.close()
    rcrispr_model = model_from_json(rcrispr_model_json)

    print("Load model weights")
    rcrispr_model.load_weights(rcrispr_weights)

    x = matrix_test
    y = label_test
    y_pred = rcrispr_model.predict(x).flatten()

    # auroc
    fpr, tpr, _ = metrics.roc_curve(y, y_pred)
    auroc = metrics.roc_auc_score(y, y_pred)

    # auprc
    prc, rec, _ = metrics.precision_recall_curve(y, y_pred)
    prc[[rec == 0]] = 1.0
    auprc = metrics.auc(rec, prc)

    print("Test data is ",test_data," AUROC is ", auroc, "AUPRC is ", auprc)

    if fig_auroc or fig_auprc:
        plt.figure(1)
        if fig_auroc:
            plt.title('Receiver Operating Characteristic Curve')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            line = plt.plot(fpr, tpr, label='AUROC = %0.3f' %auroc)
            plt.legend(labels = ['R-CRISPR AUROC = %0.3f' % auroc])
            plt.show()
            return auroc

        if fig_auprc:
            plt.title('Precision Recall Curve')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            line = plt.plot(rec, prc, label='AUPRC = %0.3f' % auprc)
            plt.legend(labels=['R-CRISPR AUPRC = %0.3f' % auprc])
            plt.show()
            return auprc

    return 0


rcrispr_json = "./train_models/rcrispr_train_on_dataC.json"
rcrispr_weights = './train_models/rcrispr_train_on_dataC.h5'
test =  test_R_CRISPR(rcrispr_json, rcrispr_weights, test_data = "GUIDE_III", fig_auprc = True)




