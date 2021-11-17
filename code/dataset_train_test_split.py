import os
import pickle as pkl
import numpy as np

import dataset_encoder



def load_training_data_A():
    # CIRCLE,PKD,PDH,GUIDE_I
    train_data_path = "./train_data/data_A.pkl"
    if os.path.exists(train_data_path):
        print("Dataset A: Combination of CIRCLE, PKD, PDH, and GUIDE_I")
        matrix, label = pkl.load(open(train_data_path, "rb"))
    else:
        circle_matrix, circle_label = dataset_encoder.encode_CIRCLE_dataset()
        pkd_matrix, pkd_label = dataset_encoder.encode_PKD_dataset()
        pdh_matrix, pdh_label = dataset_encoder.encode_PDH_dataset()
        guide_matrix, guide_label = dataset_encoder.encode_GUIDE_I_dataset()
        matrix = np.concatenate([circle_matrix, pkd_matrix, pdh_matrix, guide_matrix], axis=0)
        label = np.concatenate([circle_label, pkd_label, pdh_label, guide_label], axis=0)
        pkl.dump([matrix, label], open("./train_data/data_A.pkl", "wb"))
    matrix = matrix.reshape((len(matrix), 1, 24, 7))
    print("the size of data A is ", matrix.shape)
    return matrix, label


def load_training_data_B():
    # PKD,PDH,SITE,GUIDE_I
    train_data_path = "./train_data/data_B.pkl"
    if os.path.exists(train_data_path):
        print("Dataset B: Combination of PKD, PDH, SITE, and GUIDE_I")
        matrix, label = pkl.load(open(train_data_path,"rb"))
    else:
        pkd_matrix, pkd_label = dataset_encoder.encode_PKD_dataset()
        pdh_matrix, pdh_label = dataset_encoder.encode_PDH_dataset()
        site_matrix, site_label = dataset_encoder.encode_SITE_dataset()
        guide_matrix, guide_label = dataset_encoder.encode_GUIDE_I_dataset()
        matrix = np.concatenate([pkd_matrix, pdh_matrix, site_matrix, guide_matrix], axis = 0)
        label = np.concatenate([pkd_label, pdh_label, site_label, guide_label], axis = 0)
        pkl.dump([matrix, label], open("./train_data/data_B.pkl", "wb"))
    matrix = matrix.reshape((len(matrix), 1, 24, 7))
    print("the size of data B is ", matrix.shape)
    return matrix, label


def load_training_data_C():
    # CIRCLE
    train_data_path = "./train_data/data_C.pkl"
    if os.path.exists(train_data_path):
        print("Dataset C: Combination of CIRCLE")
        circle_matrix, circle_label = pkl.load(open(train_data_path, "rb"))
    else:
        circle_matrix, circle_label = dataset_encoder.encode_CIRCLE_dataset()
        pkl.dump([circle_matrix, circle_label], open("./train_data/data_C.pkl", "wb"))
    circle_matrix = circle_matrix.reshape((len(circle_matrix), 1, 24, 7))
    print("the size of data C is ", circle_matrix.shape)
    return circle_matrix, circle_label


def load_training_data_D():
    # CIRCLE,PKD,PDH,SITE,GUIDE_I
    train_data_path = "./train_data/data_D.pkl"
    if os.path.exists(train_data_path):
        print("Dataset D: Combination of CIRCLE, PKD, PDH, SITE, and GUIDE_I")
        matrix, label = pkl.load(open(train_data_path,"rb"))
    else:
        circle_matrix, circle_label = dataset_encoder.encode_CIRCLE_dataset()
        pkd_matrix, pkd_label = dataset_encoder.encode_PKD_dataset()
        pdh_matrix, pdh_label = dataset_encoder.encode_PDH_dataset()
        site_matrix, site_label = dataset_encoder.encode_SITE_dataset()
        guide_matrix, guide_label = dataset_encoder.encode_GUIDE_I_dataset()
        matrix = np.concatenate([circle_matrix, pkd_matrix, pdh_matrix, site_matrix, guide_matrix], axis = 0)
        label = np.concatenate([circle_label, pkd_label, pdh_label, site_label, guide_label], axis = 0)
        pkl.dump([matrix, label], open("./train_data/data_D.pkl", "wb"))
    matrix = matrix.reshape((len(matrix), 1, 24, 7))
    print("the size of data D is ", matrix.shape)
    return matrix, label


def load_training_data_E():
    # SITE
    train_data_path = "./train_data/data_E.pkl"
    if os.path.exists(train_data_path):
        print("Dataset E: Combination of SITE")
        site_matrix, site_label = pkl.load(open(train_data_path, "rb"))
    else:
        site_matrix, site_label = dataset_encoder.encode_SITE_dataset()
        pkl.dump([site_matrix, site_label], open("./train_data/data_E.pkl", "wb"))
    site_matrix = site_matrix.reshape((len(site_matrix), 1, 24, 7))
    print("the size of data E is ", site_matrix.shape)
    return site_matrix, site_label


def load_training_data_F():
    # PKD,PDH,GUIDE_I
    train_data_path = "./train_data/data_F.pkl"
    if os.path.exists(train_data_path):
        print("Dataset F: Combination of PKD, PDH, and GUIDE_I")
        matrix, label = pkl.load(open(train_data_path, "rb"))
    else:
        pkd_matrix, pkd_label = dataset_encoder.encode_PKD_dataset()
        pdh_matrix, pdh_label = dataset_encoder.encode_PDH_dataset()
        guide_matrix, guide_label = dataset_encoder.encode_GUIDE_I_dataset()
        matrix = np.concatenate([pkd_matrix, pdh_matrix, guide_matrix], axis=0)
        label = np.concatenate([pkd_label, pdh_label, guide_label], axis=0)
        pkl.dump([matrix, label], open("./train_data/data_F.pkl", "wb"))
    matrix = matrix.reshape((len(matrix), 1, 24, 7))
    print("the size of data F is ", matrix.shape)
    return matrix, label


def load_training_data_G():
    # CIRCLE, SITE
    train_data_path = "./train_data/data_G.pkl"
    if os.path.exists(train_data_path):
        print("Dataset G: Combination of CIRCLE and SITE")
        matrix, label = pkl.load(open(train_data_path, "rb"))
    else:
        circle_matrix, circle_label = dataset_encoder.encode_CIRCLE_dataset()
        site_matrix, site_label = dataset_encoder.encode_SITE_dataset()
        matrix = np.concatenate([circle_matrix, site_matrix], axis=0)
        label = np.concatenate([circle_label, site_label], axis=0)
        pkl.dump([matrix, label], open("./train_data/data_G.pkl", "wb"))
    matrix = matrix.reshape((len(matrix), 1, 24, 7))
    print("the size of data G is ", matrix.shape)
    return matrix, label


def load_testing_data_GUIDE_II():
    # GUIDE_II
    train_data_path = "./test_data/data_GUIDE_II.pkl"
    if os.path.exists(train_data_path):
        print("Dataset GUIDE_II")
        guide_matrix, guide_label = pkl.load(open(train_data_path, "rb"))
    else:
        guide_matrix, guide_label = dataset_encoder.encode_GUIDE_II_dataset()
        pkl.dump([guide_matrix, guide_label], open("./test_data/data_GUIDE_II.pkl", "wb"))
    guide_matrix = guide_matrix.reshape((len(guide_matrix), 1, 24, 7))
    print("the size of data GUIDE_II is ", guide_matrix.shape)
    return guide_matrix, guide_label


def load_testing_data_GUIDE_III():
    train_data_path = "./test_data/data_GUIDE_III.pkl"
    if os.path.exists(train_data_path):
        print("Dataset GUIDE_III")
        guide_matrix, guide_label = pkl.load(open(train_data_path, "rb"))
    else:
        guide_matrix, guide_label = dataset_encoder.encode_GUIDE_III_dataset()
        pkl.dump([guide_matrix, guide_label], open("./test_data/data_GUIDE_III.pkl", "wb"))
    guide_matrix = guide_matrix.reshape((len(guide_matrix), 1, 24, 7))
    print("the size of data GUIDE_III is ", guide_matrix.shape)
    return guide_matrix, guide_label


def load_training_data_PDH():
    train_data_path = "./train_data/data_pdh.pkl"
    if os.path.exists(train_data_path):
        print("Dataset F: Combination of PKD, PDH, and GUIDE_I")
        matrix, label = pkl.load(open(train_data_path, "rb"))
    else:
        pdh_matrix, pdh_label = dataset_encoder.encode_PDH_dataset()
        pkl.dump([pdh_matrix, pdh_label], open("./train_data/data_pdh.pkl", "wb"))
        label = pdh_label
        matrix = pdh_matrix
    matrix = matrix.reshape((len(matrix), 1, 24, 7))
    print("the size of data F is ", matrix.shape)
    return matrix, label




