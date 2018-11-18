#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
#
# Copyright (c) 2018 Wan Li. All Rights Reserved
#
########################################################################

"""
File: demo.py
Author: leowan(leowan)
Date: 2018/11/16 16:14:36
"""

import os
import shutil

import numpy as np
from scipy import sparse
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix

import tflrclassifier
import tflrclassifier_column_indexed
import utils

def test_dense(dataset):
    """
        Unittest dense predict
    """
    tflr = tflrclassifier.TFLRClassifier(batch_size=1, print_step=100, input_type='dense')
    X = dataset[0]
    y = dataset[1]
    tflr.fit(X, y)
    logits = tflr.decision_function(X)
    pred = tflr.predict(X)
    print(accuracy_score(y, pred))
    print(confusion_matrix(y, pred))
    print(roc_auc_score(y, logits))

def test_sparse(dataset):
    """
        Unittest dense predict
    """
    tflr = tflrclassifier.TFLRClassifier(batch_size=1, print_step=100, input_type='sparse')
    X = sparse.csr_matrix(dataset[0])
    y = dataset[1]
    tflr.fit(X, y)
    logits = tflr.decision_function(X)
    pred = tflr.predict(X)
    print(accuracy_score(y, pred))
    print(confusion_matrix(y, pred))
    print(roc_auc_score(y, logits))

def test_load_chkpt(dataset):
    """
        Unittest load checkpoint
    """
    chkpt_dir = './tmp'
    if os.path.exists(chkpt_dir):
        shutil.rmtree(chkpt_dir)

    tflr = tflrclassifier.TFLRClassifier(batch_size=1, input_type='dense',
        chkpt_dir=chkpt_dir)
    X = dataset[0]
    y = dataset[1]
    tflr.fit(X, y)
    logits = tflr.decision_function(X)
    pred = tflr.predict(X)
    print('train acc: {}'.format(accuracy_score(y, pred)))
    lr_load = tflrclassifier.TFLRClassifier(input_type='dense')
    lr_load.load_checkpoint(feature_num=X.shape[1], chkpt_dir=chkpt_dir)
    print('loaded acc: {}'.format(accuracy_score(y, lr_load.predict(X))))

def test_export(dataset, input_type='dense'):
    """
        Unittest import / export Pb model
    """
    export_dir = './tmp'
    if os.path.exists(export_dir):
        shutil.rmtree(export_dir)
    tflr = tflrclassifier.TFLRClassifier(batch_size=1, input_type=input_type)
    X = dataset[0]
    if input_type == 'sparse':
        X = sparse.csr_matrix(X)
    y = dataset[1]
    tflr.fit(X, y)
    logits = tflr.decision_function(X)
    print('train auc: {}'.format(roc_auc_score(y, logits)))
    tflr.export_model(export_dir, input_type)
    logits = tflr.decision_function_imported(X, export_dir, input_type)
    print('loaded auc: {}'.format(roc_auc_score(y, logits)))

def test_colind_model(X_colind, X_colval, y, feature_num):
    """
        Unittest column-indexed model
    """
    model = tflrclassifier_column_indexed.TFLRClassifier(
        feature_num=feature_num, # feature num must set
        l2_weight=0.01, learning_rate=1e-2,
        batch_size=1, epoch_num=10, print_step=1000, random_seed=42)
    model.fit(np.array(X_ind_tr), np.array(X_val_tr), np.array(y_cid_tr))
    predictions = model.predict(np.array(X_ind_tr), np.array(X_val_tr))
    print('model: {}'.format(model.__str__()))
    print('train acc: {}'.format(accuracy_score(
        np.array(y_cid_tr), model.predict(np.array(X_ind_tr), np.array(X_val_tr)))))

def test_colind_load_chkpt(X_colind, X_colval, y, feature_num):
    """
        Unittest load checkpoint
    """
    chkpt_dir = './tmp'
    if os.path.exists(chkpt_dir):
        shutil.rmtree(chkpt_dir)

    model = tflrclassifier_column_indexed.TFLRClassifier(
        feature_num=feature_num, # feature num must set
        l2_weight=0.01, learning_rate=1e-2,
        batch_size=1, epoch_num=10, print_step=1000, random_seed=42, chkpt_dir=chkpt_dir)
    model.fit(np.array(X_ind_tr), np.array(X_val_tr), np.array(y_cid_tr))
    print('train acc: {}'.format(accuracy_score(
        np.array(y_cid_tr), model.predict(np.array(X_ind_tr), np.array(X_val_tr)))))
    model_load = tflrclassifier_column_indexed.TFLRClassifier(
        feature_num=feature_num, # feature num must set
        l2_weight=0.01, learning_rate=1e-2,
        batch_size=10, epoch_num=10, print_step=1000, random_seed=42)
    model_load.load_checkpoint(feature_num=feature_num, chkpt_dir=chkpt_dir)
    print('loaded acc: {}'.format(accuracy_score(
        np.array(y_cid_tr), model_load.predict(np.array(X_ind_tr), np.array(X_val_tr)))))

def test_colind_export(X_colind, X_colval, y, feature_num):
    """
        Unittest import / export Pb model
    """
    export_dir = './tmp'
    if os.path.exists(export_dir):
        shutil.rmtree(export_dir)

    model = tflrclassifier_column_indexed.TFLRClassifier(
        feature_num=feature_num, # feature num must set
        l2_weight=0.01, learning_rate=1e-2,
        batch_size=1, epoch_num=10, print_step=1000, random_seed=42)
    model.fit(np.array(X_ind_tr), np.array(X_val_tr), np.array(y_cid_tr))
    print('train auc: {}'.format(roc_auc_score(
        np.array(y_cid_tr), model.decision_function(np.array(X_ind_tr), np.array(X_val_tr)))))
    model.export_model(export_dir)
    print('loaded auc: {}'.format(roc_auc_score(
        np.array(y_cid_tr), model.decision_function_imported(
        np.array(X_ind_tr), np.array(X_val_tr), import_dir=export_dir))))

if __name__ == "__main__":
    # iris data
    from sklearn import datasets
    dataset_ori = datasets.load_iris(return_X_y=True)
    y_label = map(lambda x: x == 0, dataset_ori[1])
    dataset = []
    dataset.append(dataset_ori[0])
    dataset.append(np.array(list(y_label)).astype(int))

    test_dense(dataset)
    test_sparse(dataset)
    test_load_chkpt(dataset)
    test_export(dataset, input_type='dense')
    test_export(dataset, input_type='sparse')

    # test column-indexed model
    from sklearn.datasets import dump_svmlight_file
    from sklearn.model_selection import train_test_split
    fname = './dump_svmlight.txt'
    feature_num = dataset[0].shape[1]
    dump_svmlight_file(dataset[0], dataset[1], fname)
    X_cid_tr, y_cid_tr = utils.read_zipped_column_indexed_data_from_svmlight_file(fname)
    X_ind_tr, X_val_tr, y_cid_tr = utils.convert_to_column_indexed_data(X_cid_tr, y_cid_tr)
    X_ind_tr, X_val_tr, y_cid_tr = utils.convert_to_fully_column_indexed_data(
        X_ind_tr, X_val_tr, y_cid_tr, feature_num=feature_num)

    test_colind_model(X_ind_tr, X_val_tr, y_cid_tr, feature_num)
    test_colind_load_chkpt(X_ind_tr, X_val_tr, y_cid_tr, feature_num)
    test_colind_export(X_ind_tr, X_val_tr, y_cid_tr, feature_num)
