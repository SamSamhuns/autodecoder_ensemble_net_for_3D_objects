# General
import os
import sys
import time
import math
import tqdm
import random
import numpy as np
import matplotlib.pylab as plt

# Torch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

# sklearn
import joblib
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import hinge_loss, log_loss
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import auc, roc_curve

# custom imports
from .datasets import PointDriftDS, EncodingDS, EncodingDS
from .models import AutoDecoder, EnsembleAutoDecoder, CompNet, EnsembleCompNet
from .utils import chamfer_loss, visualize_npy, plot_roc_curve, print_model_metrics, HyperParameter, DirectorySetting


def find_encoding(X, y, autodecoder, encoding_iters=300,
                  encoding_size=256, lr=5e-4, l2_reg=False):
    """
    Generate the encoding (latent vector) for each data in X
    """

    def _adjust_lr(initial_lr, optimizer, num_iters, decreased_by, adjust_lr_every):

        lr = initial_lr * ((1 / decreased_by) **
                           (num_iters // adjust_lr_every))
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    decreased_by = 10
    adjust_lr_every = encoding_iters // 2

    encoding = torch.ones(X.shape[0], encoding_size).normal_(
        mean=0, std=1.0 / math.sqrt(encoding_size)).cuda()

    encoding.requires_grad = True
    optimizer = torch.optim.Adam([encoding], lr=lr)
    loss_num = 0

    for i in range(encoding_iters):
        autodecoder.eval()
        _adjust_lr(lr, optimizer, i, decreased_by, adjust_lr_every)
        optimizer.zero_grad()
        y_pred = autodecoder(X, encoding)
        loss = chamfer_loss(y_pred, y, ps=y.shape[-1])

        if l2_reg:
            loss += 1e-4 * torch.mean(encoding.pow(2))
        loss.backward()
        optimizer.step()

        if i % 50 == 0:
            print(i, loss.cpu().data.numpy(), encoding.norm())
        loss_num = loss.cpu().data.numpy()

    return loss_num, encoding


def train_compnet(HP, DS, train_ds, test_ds=None, compnet=None, save_wt_fname='pnet_compnet.pth'):
    """
    Train the CompNet

    Suggested Parameters
    EPOCHS=10
    batch_size=16
    encoding_size=256
    learning_rate=0.001
    """
    EPOCHS = HP.epochs
    point_dim = HP.num_point_cloud
    batch_size = HP.batch_size
    encoding_size = HP.encoding_size
    lr = HP.learning_rate

    if compnet is None:
        cpnet = CompNet(encoding_size=encoding_size)
    else:
        cpnet = compnet
    cpnet = cpnet.cuda()

    optimizer = torch.optim.Adam(cpnet.parameters(), lr=lr)
    op_schedule = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    cpnet = nn.DataParallel(cpnet)
    cpnet.cuda()

    data_loader_train = DataLoader(train_ds, batch_size=batch_size,
                                   shuffle=True)

    loss_fn = nn.BCELoss()
    for epoch in range(EPOCHS):
        same_total_loss = 0.0
        diff_total_loss = 0.0
        cpnet.train()
        for batch_idx, (x, y, z, idx, same_cls, diff_cls) in enumerate(data_loader_train):
            optimizer.zero_grad()

            same_cls, diff_cls = same_cls.cuda(), diff_cls.cuda()
            same_cls, diff_cls = Variable(
                same_cls).float(), Variable(diff_cls).float()

            same_pred = cpnet(same_cls)
            same_target = torch.ones(same_pred.shape).float().cuda()
            same_loss = loss_fn(same_pred, same_target)
            same_loss.backward()
            same_total_loss += same_loss.data.cpu().numpy()

            diff_pred = cpnet(diff_cls)
            diff_target = torch.zeros(diff_pred.shape).float().cuda()
            diff_loss = loss_fn(diff_pred, diff_target)
            diff_loss.backward()
            diff_total_loss += diff_loss.data.cpu().numpy()

            optimizer.step()
            if batch_idx % 100 == 0 and batch_idx > 0:
                print(f"Epoch: {epoch}. batch_idx: {batch_idx}")
                print("Loss: ", same_total_loss / 100, diff_total_loss / 100)
                same_total_loss = 0.0
                diff_total_loss = 0.0
        op_schedule.step(epoch)

        if test_ds is not None and epoch % 5 == 0:
            print("Eval: ", eval_compnet(cpnet, test_ds, batch_size=batch_size))

    if save_wt_fname is not None:
        torch.save(cpnet.module.state_dict(),
                   DS.CLASSIFIER_TRAINED_WEIGHT_DIR + '/' + save_wt_fname)
    return cpnet


def get_compnet_y_test_and_y_score(cpnet, test_ds, batch_size=16):
    """
    Returns y_test, y_score given a NN compnet and the test encoding ds
    """
    cpnet.eval()
    test_dl = DataLoader(test_ds, batch_size=batch_size,
                         shuffle=False)

    y_test, y_score = [], []
    for batch_idx, (_, _, _, _, same_cls, diff_cls) in enumerate(test_dl):
        same_cls, diff_cls = same_cls.cuda(), diff_cls.cuda()

        same_pred = cpnet(same_cls).detach().cpu().numpy()
        same_target = np.ones(same_pred.shape, dtype=np.float)
        
        y_test.extend(same_target)
        y_score.extend(same_pred)

        diff_pred = cpnet(diff_cls).detach().cpu().numpy()
        diff_target = np.zeros(diff_pred.shape, dtype=np.float)
        
        y_test.extend(diff_target)
        y_score.extend(diff_pred)
        
    return y_test, y_score


def eval_compnet(cpnet, test_ds, batch_size=16, pred_threshold=0.5):
    cpnet.eval()
    test_dl = DataLoader(test_ds, batch_size=batch_size,
                         shuffle=False)
    loss_fn = nn.BCELoss()

    same_total_loss = 0.0
    diff_total_loss = 0.0
    batch_cnt = 0
    same_corr_cnt = 0.0
    diff_corr_cnt = 0.0
    same_incorr_cnt = 0.0
    diff_incorr_cnt = 0.0


    for batch_idx, (x, y, z, idx, same_cls, diff_cls) in enumerate(test_dl):
        batch_cnt += 1
        same_cls, diff_cls = same_cls.cuda(), diff_cls.cuda()

        same_pred = cpnet(same_cls)
        same_target = torch.ones(same_pred.shape).float().cuda()
        same_loss = loss_fn(same_pred, same_target)
        same_total_loss += same_loss.data.cpu().numpy()

        same_corr_cnt += np.sum(same_pred.detach().cpu().numpy()
                                > pred_threshold)
        same_incorr_cnt += np.sum(same_pred.detach().cpu().numpy()
                                  <= pred_threshold)

        diff_pred = cpnet(diff_cls)
        diff_target = torch.zeros(diff_pred.shape).float().cuda()
        diff_loss = loss_fn(diff_pred, diff_target)
        diff_total_loss += diff_loss.data.cpu().numpy()

        diff_corr_cnt += np.sum(diff_pred.detach().cpu().numpy()
                                < pred_threshold)
        diff_incorr_cnt += np.sum(diff_pred.detach().cpu().numpy()
                                  >= pred_threshold)

    precision = same_corr_cnt / (same_corr_cnt+diff_incorr_cnt)
    # same_corr_cnt / len(test_ds)
    recall = same_corr_cnt / (same_corr_cnt+same_incorr_cnt)
    print("------------------ Evaluation Report ------------------")
    print(f"Total Accuracy: {(same_corr_cnt+diff_corr_cnt)/(2*len(test_ds))}")
    print(f"After {batch_cnt} batches and {len(test_ds)} test points")
    print()

    print(f"Metrics for the same class:")
    print(f"Avg loss: {same_total_loss / batch_cnt}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {(2*precision*recall)/(precision+recall)}")

    precision = diff_corr_cnt / (diff_corr_cnt+same_incorr_cnt)
    # diff_corr_cnt / len(test_ds)
    recall = diff_corr_cnt / (diff_corr_cnt+diff_incorr_cnt)
    print()
    print(f"Metrics for the diff class:")
    print(f"Avg loss: {diff_total_loss / batch_cnt}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {(2*precision*recall)/(precision+recall)}")
    
    # Get AUC and plot ROC
    y_true, y_score = get_y_test_and_y_score(cpnet, test_ds)
    fp_rate, tp_rate, _ = roc_curve(y_true, y_score)
    plot_roc_curve(fp_rate, tp_rate, title='CompNet Receiver operating characteristic')

    return (same_total_loss, diff_total_loss,
            same_corr_cnt, diff_corr_cnt,
            same_incorr_cnt, diff_incorr_cnt,
            batch_cnt, len(test_ds))


def train_decoder(HP, DS, train_ds, test_ds=None, decoder=None, save_wt_fname='pnet_decoder.pth'):
    """ 
    Default training is for 3D point dimensions

    Suggested Settings
        EPOCHS = 10
        point_dim = 3
        batch_size = 16
        learning_rate = 0.001
        encoding_size = 256

    Set save_wt_fname to None to disable weight saves
    """
    EPOCHS = HP.epochs
    point_dim = HP.num_point_cloud
    batch_size = HP.batch_size
    encoding_size = HP.encoding_size
    lr = HP.learning_rate

    if decoder is None:
        adnet = AutoDecoder(encoding_size, point_dim)
    else:
        adnet = decoder
    adnet = adnet.cuda()

    # encodings for same class transformation
    same_encoding = torch.nn.Embedding(
        len(train_ds), encoding_size, max_norm=1.0)
    # init encoding with Kaiming Initialization
    torch.nn.init.normal_(same_encoding.weight.data,
                          0.0,
                          1.0 / math.sqrt(encoding_size))

    # encodings for different class transformation
    diff_encoding = torch.nn.Embedding(
        len(train_ds), encoding_size, max_norm=1.0)
    # init encoding with Kaiming Initialization
    torch.nn.init.normal_(diff_encoding.weight.data,
                          0.0,
                          1.0 / math.sqrt(encoding_size))

    optimizer = torch.optim.Adam([
        {"params": adnet.parameters(), "lr": lr, },
        {"params": same_encoding.parameters(), "lr": lr, },
        {"params": diff_encoding.parameters(), "lr": lr, }, ])

    op_schedule = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    adnet = nn.DataParallel(adnet)
    adnet.cuda()

    data_loader_train = DataLoader(train_ds, batch_size=batch_size,
                                   shuffle=True)

    for epoch in range(0, EPOCHS):
        adnet.train()
        same_total_loss = 0.0
        diff_total_loss = 0.0

        for batch_idx, (x, y, z, idx) in enumerate(data_loader_train):
            optimizer.zero_grad()
            x, y, z = x.cuda(), y.cuda(), z.cuda()
            x, y, z = (Variable(x).float(),
                       Variable(y).float(),
                       Variable(z).float())
            y_pred = adnet(x, same_encoding(torch.LongTensor(idx)))
            loss_cham = chamfer_loss(y, y_pred, ps=y.shape[-1])
            same_total_loss += loss_cham.data.cpu().numpy()
            loss_cham.backward()

            z_pred = adnet(x, diff_encoding(torch.LongTensor(idx)))
            loss_cham = chamfer_loss(z, z_pred, ps=z.shape[-1])
            diff_total_loss += loss_cham.data.cpu().numpy()
            loss_cham.backward()

            optimizer.step()
            if batch_idx % 100 == 0 and batch_idx > 0:
                print(f"Epoch: {epoch}. batch_idx: {batch_idx}")
                print("Loss: ", same_total_loss / 100, diff_total_loss / 100)
                same_total_loss = 0.0
                diff_total_loss = 0.0
        op_schedule.step(epoch)

        if test_ds is not None and epoch % 5 == 0:
            print("Eval: ", eval_decoder(adnet, test_ds, batch_size=batch_size))

    if save_wt_fname is not None:
        torch.save(adnet.module.state_dict(),
                   DS.AUTODECODER_TRAINED_WEIGHT_DIR + '/' + save_wt_fname)
    return adnet


def return_decoder_train_test_encoding_ds(X_train, y_train, X_test, y_test,
                                          adnet_HP, adnet_DS,
                                          autodecoder=AutoDecoder(encoding_size=256,
                                                                  point_dim=3),
                                          save_wt_fname='mnet_decoder.pth'):
    """
    Trains the decoder by generating the same and different class datasets
    Return Value: autodecoder, train_encoding_ds, test_encoding_ds, train_ds, test_ds

    Suggested adnet_HP:
        adnet_HP = HyperParameter(lr=0.001, epochs=4)
    """

    train_ds = PointDriftDS(X_train, y_train)
    test_ds = PointDriftDS(X_test, y_test)

    # train autodecoder
    mn_autodecoder = train_decoder(adnet_HP,
                                   adnet_DS,
                                   train_ds=train_ds,
                                   test_ds=test_ds,
                                   decoder=autodecoder,
                                   save_wt_fname=save_wt_fname)

    # get the train encodings
    train_encoding_ds = EncodingDS(PointDriftDS(
        X_train, y_train), mn_autodecoder)
    train_result = train_encoding_ds.train_encodings(
        find_encoding,
        num_iterations=15, lr=0.05)

    # get the test encodings
    test_encoding_ds = EncodingDS(PointDriftDS(
        X_test, y_test), mn_autodecoder)
    test_result = test_encoding_ds.train_encodings(
        find_encoding,
        num_iterations=15, lr=0.05)

    return autodecoder, train_encoding_ds, test_encoding_ds, train_ds, test_ds


def eval_decoder(decoder, eval_ds, batch_size=16):
    decoder.eval()
    encoding_ds = EncodingDS(eval_ds, decoder)
    return encoding_ds.train_encodings(find_encoding,
                                       num_iterations=10, lr=0.05, batch_size=batch_size)[2:]


def train_rand_forest(HP, DS, train_ds, save_wt_fname='pnet_rand_forest_clf.pkl', **kwargs):
    """
    Train the Random Forest model

    **kwargs send key word arguments to the Decision Trees Classifier

    Suggested Parameters
    HP.max_depth = 2
    HP.batch_size = 16
    """
    random_state_seed = HP.seed
    max_depth = HP.max_depth
    criterion = HP.criterion
    n_estimators = HP.n_estimators
    min_samples_split = HP.min_samples_split

    data_loader_train = DataLoader(train_ds, batch_size=16,
                                   shuffle=True)
    X, y = None, None
    # Combine the entire dataset
    for batch_idx, (_x, _y, _z, _idx, same_cls, diff_cls) in enumerate(data_loader_train):
        same_cls, diff_cls = same_cls.detach().numpy(), diff_cls.detach().numpy()
        same_target, diff_target = np.ones(
            same_cls.shape[0]), np.zeros(diff_cls.shape[0])

        if X is None and y is None:
            X = np.concatenate([same_cls, diff_cls], axis=0)
            y = np.concatenate([same_target, diff_target], axis=0)
        else:
            X = np.concatenate([X, same_cls, diff_cls], axis=0)
            y = np.concatenate([y, same_target, diff_target], axis=0)

    rand_forest_clf = RandomForestClassifier(max_depth=max_depth,
                                             criterion=criterion,
                                             n_estimators=n_estimators,
                                             random_state=random_state_seed,
                                             min_samples_split=min_samples_split)

    rand_forest_clf.fit(X, y)
    y_pred = rand_forest_clf.predict(X)

    print(f"Total Log Loss: {log_loss(y, rand_forest_clf.predict_proba(X))}")
    if save_wt_fname is not None:
        _ = joblib.dump(rand_forest_clf, DS.CLASSIFIER_TRAINED_WEIGHT_DIR+'/'+save_wt_fname,
                        compress=9)
    return rand_forest_clf


def eval_rand_forest(rand_forest_clf, test_ds, batch_size=16):
    test_dl = DataLoader(test_ds, batch_size=batch_size,
                         shuffle=False)
    X, y = None, None
    # Combine the entire dataset
    for batch_idx, (_x, _y, _z, _idx, same_cls, diff_cls) in enumerate(test_dl):
        same_cls, diff_cls = same_cls.detach().numpy(), diff_cls.detach().numpy()
        same_target, diff_target = np.ones(
            same_cls.shape[0]), np.zeros(diff_cls.shape[0])

        if X is None and y is None:
            X = np.concatenate([same_cls, diff_cls], axis=0)
            y = np.concatenate([same_target, diff_target], axis=0)
        else:
            X = np.concatenate([X, same_cls, diff_cls], axis=0)
            y = np.concatenate([y, same_target, diff_target], axis=0)

    """
    Note:
    rbf_feature reduces dimensions of training data
    but also penalizes accuracy, precision and recall of model
        rbf_feature = RBFSampler(gamma=1, random_state=1)
        X = rbf_feature.fit_transform(X)
    """
    y_pred = rand_forest_clf.predict(X)
    y, y_pred = y.astype(int), y_pred.astype(int)

    same_corr_cnt = np.sum(y_pred & y)
    same_incorr_cnt = np.sum(y & (y_pred ^ 1))
    diff_corr_cnt = np.sum((y ^ 1) & (y_pred ^ 1))
    diff_incorr_cnt = np.sum((y ^ 1) & y_pred)

    total_loss = log_loss(y, rand_forest_clf.predict_proba(X))
    
    print_model_metrics(total_loss, same_corr_cnt, same_incorr_cnt,
                        diff_corr_cnt, diff_incorr_cnt, len(test_ds))
    
    # Get AUC and plot ROC
    y_true, y_score = y, rand_forest_clf.predict_proba(X)
    fp_rate, tp_rate, _ = roc_curve(y_true, y_score)
    plot_roc_curve(fp_rate, tp_rate, title='Random Forest Receiver operating characteristic')

    return (total_loss,
            same_corr_cnt, diff_corr_cnt,
            same_incorr_cnt, diff_incorr_cnt,
            len(test_ds))


def train_log_regr(HP, DS, train_ds, save_wt_fname='pnet_log_regr_clf.pkl', **kwargs):
    """
    Train the Logistic Regression model

    **kwargs send key word arguments to the Decision Trees Classifier

    Suggested Parameters
    HP.batch_size=16
    """
    solver = HP.solver
    data_loader_train = DataLoader(train_ds, batch_size=16,
                                   shuffle=True)
    X, y = None, None
    # Combine the entire dataset
    for batch_idx, (_x, _y, _z, _idx, same_cls, diff_cls) in enumerate(data_loader_train):
        same_cls, diff_cls = same_cls.detach().numpy(), diff_cls.detach().numpy()
        same_target, diff_target = np.ones(
            same_cls.shape[0]), np.zeros(diff_cls.shape[0])

        if X is None and y is None:
            X = np.concatenate([same_cls, diff_cls], axis=0)
            y = np.concatenate([same_target, diff_target], axis=0)
        else:
            X = np.concatenate([X, same_cls, diff_cls], axis=0)
            y = np.concatenate([y, same_target, diff_target], axis=0)

    lor_regr_clf = LogisticRegression(random_state=0, solver=solver)

    lor_regr_clf.fit(X, y)
    y_pred = lor_regr_clf.predict(X)

    print(f"Total Log Loss: {log_loss(y, lor_regr_clf.predict_proba(X))}")
    if save_wt_fname is not None:
        _ = joblib.dump(lor_regr_clf, DS.CLASSIFIER_TRAINED_WEIGHT_DIR+'/'+save_wt_fname,
                        compress=9)
    return lor_regr_clf


def eval_log_regr(log_regr_clf, test_ds, batch_size=16):
    test_dl = DataLoader(test_ds, batch_size=batch_size,
                         shuffle=False)
    X, y = None, None
    # Combine the entire dataset
    for batch_idx, (_x, _y, _z, _idx, same_cls, diff_cls) in enumerate(test_dl):
        same_cls, diff_cls = same_cls.detach().numpy(), diff_cls.detach().numpy()
        same_target, diff_target = np.ones(
            same_cls.shape[0]), np.zeros(diff_cls.shape[0])

        if X is None and y is None:
            X = np.concatenate([same_cls, diff_cls], axis=0)
            y = np.concatenate([same_target, diff_target], axis=0)
        else:
            X = np.concatenate([X, same_cls, diff_cls], axis=0)
            y = np.concatenate([y, same_target, diff_target], axis=0)

    """
    Note:
    rbf_feature reduces dimensions of training data
    but also penalizes accuracy, precision and recall of model
        rbf_feature = RBFSampler(gamma=1, random_state=1)
        X = rbf_feature.fit_transform(X)
    """
    y_pred = log_regr_clf.predict(X)
    y, y_pred = y.astype(int), y_pred.astype(int)

    same_corr_cnt = np.sum(y_pred & y)
    same_incorr_cnt = np.sum(y & (y_pred ^ 1))
    diff_corr_cnt = np.sum((y ^ 1) & (y_pred ^ 1))
    diff_incorr_cnt = np.sum((y ^ 1) & y_pred)

    total_loss = log_loss(y, log_regr_clf.predict_proba(X))

    print_model_metrics(total_loss, same_corr_cnt, same_incorr_cnt,
                        diff_corr_cnt, diff_incorr_cnt, len(test_ds))
    
    # Get AUC and plot ROC
    y_true, y_score = y, log_regr_clf.predict_proba(X)
    fp_rate, tp_rate, _ = roc_curve(y_true, y_score)
    plot_roc_curve(fp_rate, tp_rate, title='Logistic Regression Receiver operating characteristic')

    return (total_loss,
            same_corr_cnt, diff_corr_cnt,
            same_incorr_cnt, diff_incorr_cnt,
            len(test_ds))


def train_naive_bayes(HP, DS, train_ds, save_wt_fname='pnet_gaussian_naive_bayes_clf.pkl', **kwargs):
    """
    Train the Gaussian Naive Bayes Classifier

    **kwargs send key word arguments to the  Naive Bayes Classifier

    Suggested Parameters
    HP.batch_size=16
    """
    batch_size = HP.batch_size

    data_loader_train = DataLoader(train_ds, batch_size=16,
                                   shuffle=True)
    X, y = None, None
    # Combine the entire dataset
    for batch_idx, (_x, _y, _z, _idx, same_cls, diff_cls) in enumerate(data_loader_train):
        same_cls, diff_cls = same_cls.detach().numpy(), diff_cls.detach().numpy()
        same_target, diff_target = np.ones(
            same_cls.shape[0]), np.zeros(diff_cls.shape[0])

        if X is None and y is None:
            X = np.concatenate([same_cls, diff_cls], axis=0)
            y = np.concatenate([same_target, diff_target], axis=0)
        else:
            X = np.concatenate([X, same_cls, diff_cls], axis=0)
            y = np.concatenate([y, same_target, diff_target], axis=0)

    gau_nb_clf = GaussianNB()

    gau_nb_clf.fit(X, y)
    y_pred = gau_nb_clf.predict(X)

    print(
        f"Total Loss: {log_loss(y, gau_nb_clf.predict_proba(X))}")
    if save_wt_fname is not None:
        _ = joblib.dump(gau_nb_clf, DS.CLASSIFIER_TRAINED_WEIGHT_DIR+'/'+save_wt_fname,
                        compress=9)
    return gau_nb_clf


def eval_gaussian_naive_bayes(gau_nb_clf, test_ds, batch_size=16):
    test_dl = DataLoader(test_ds, batch_size=batch_size,
                         shuffle=False)
    X, y = None, None
    # Combine the entire dataset
    for batch_idx, (_x, _y, _z, _idx, same_cls, diff_cls) in enumerate(test_dl):
        same_cls, diff_cls = same_cls.detach().numpy(), diff_cls.detach().numpy()
        same_target, diff_target = np.ones(
            same_cls.shape[0]), np.zeros(diff_cls.shape[0])

        if X is None and y is None:
            X = np.concatenate([same_cls, diff_cls], axis=0)
            y = np.concatenate([same_target, diff_target], axis=0)
        else:
            X = np.concatenate([X, same_cls, diff_cls], axis=0)
            y = np.concatenate([y, same_target, diff_target], axis=0)

    """
    Note:
    rbf_feature reduces dimensions of training data
    but also penalizes accuracy, precision and recall of model
        rbf_feature = RBFSampler(gamma=1, random_state=1)
        X = rbf_feature.fit_transform(X)
    """
    y_pred = gau_nb_clf.predict(X)
    y, y_pred = y.astype(int), y_pred.astype(int)

    same_corr_cnt = np.sum(y_pred & y)
    same_incorr_cnt = np.sum(y & (y_pred ^ 1))
    diff_corr_cnt = np.sum((y ^ 1) & (y_pred ^ 1))
    diff_incorr_cnt = np.sum((y ^ 1) & y_pred)

    total_loss = log_loss(y, gau_nb_clf.predict_proba(X))

    print_model_metrics(total_loss, same_corr_cnt, same_incorr_cnt,
                        diff_corr_cnt, diff_incorr_cnt, len(test_ds))
    
    # Get AUC and plot ROC
    y_true, y_score = y, gau_nb_clf.predict_proba(X)
    fp_rate, tp_rate, _ = roc_curve(y_true, y_score)
    plot_roc_curve(fp_rate, tp_rate, title='Naive Bayes Receiver operating characteristic')

    return (total_loss,
            same_corr_cnt, diff_corr_cnt,
            same_incorr_cnt, diff_incorr_cnt,
            len(test_ds))


def train_decision_trees(HP, DS, train_ds, save_wt_fname='pnet_decision_trees_clf.pkl', **kwargs):
    """
    Train the Decision Tree

    **kwargs send key word arguments to the Decision Trees Classifier

    Suggested Parameters
    HP.criterion='entropy'
    HP.min_samples_split = 5 
    HP.max_features='auto' # sqrt(n_features) considered when splitting
    HP.batch_size=16
    """
    batch_size = HP.batch_size
    criterion = HP.criterion
    min_samples_split = HP.min_samples_split
    max_features = HP.max_features
    data_loader_train = DataLoader(train_ds, batch_size=16,
                                   shuffle=True)
    X, y = None, None
    # Combine the entire dataset
    for batch_idx, (_x, _y, _z, _idx, same_cls, diff_cls) in enumerate(data_loader_train):
        same_cls, diff_cls = same_cls.detach().numpy(), diff_cls.detach().numpy()
        same_target, diff_target = np.ones(
            same_cls.shape[0]), np.zeros(diff_cls.shape[0])

        if X is None and y is None:
            X = np.concatenate([same_cls, diff_cls], axis=0)
            y = np.concatenate([same_target, diff_target], axis=0)
        else:
            X = np.concatenate([X, same_cls, diff_cls], axis=0)
            y = np.concatenate([y, same_target, diff_target], axis=0)

    decision_trees_clf = tree.DecisionTreeClassifier(criterion=criterion,
                                                     min_samples_split=min_samples_split,
                                                     max_features=max_features,
                                                     **kwargs)

    decision_trees_clf.fit(X, y)
    y_pred = decision_trees_clf.predict(X)

    print(
        f"Total Log Loss: {log_loss(y, decision_trees_clf.predict_proba(X))}")
    if save_wt_fname is not None:
        _ = joblib.dump(decision_trees_clf, DS.CLASSIFIER_TRAINED_WEIGHT_DIR+'/'+save_wt_fname,
                        compress=9)
    return decision_trees_clf


def eval_decision_trees(decision_trees_clf, test_ds, batch_size=16):
    test_dl = DataLoader(test_ds, batch_size=batch_size,
                         shuffle=False)
    X, y = None, None
    # Combine the entire dataset
    for batch_idx, (_x, _y, _z, _idx, same_cls, diff_cls) in enumerate(test_dl):
        same_cls, diff_cls = same_cls.detach().numpy(), diff_cls.detach().numpy()
        same_target, diff_target = np.ones(
            same_cls.shape[0]), np.zeros(diff_cls.shape[0])

        if X is None and y is None:
            X = np.concatenate([same_cls, diff_cls], axis=0)
            y = np.concatenate([same_target, diff_target], axis=0)
        else:
            X = np.concatenate([X, same_cls, diff_cls], axis=0)
            y = np.concatenate([y, same_target, diff_target], axis=0)

    """
    Note:
    rbf_feature reduces dimensions of training data
    but also penalizes accuracy, precision and recall of model
        rbf_feature = RBFSampler(gamma=1, random_state=1)
        X = rbf_feature.fit_transform(X)
    """
    y_pred = decision_trees_clf.predict(X)
    y, y_pred = y.astype(int), y_pred.astype(int)

    same_corr_cnt = np.sum(y_pred & y)
    same_incorr_cnt = np.sum(y & (y_pred ^ 1))
    diff_corr_cnt = np.sum((y ^ 1) & (y_pred ^ 1))
    diff_incorr_cnt = np.sum((y ^ 1) & y_pred)

    total_loss = log_loss(y, decision_trees_clf.predict_proba(X))

    print_model_metrics(total_loss, same_corr_cnt, same_incorr_cnt,
                        diff_corr_cnt, diff_incorr_cnt, len(test_ds))
    
    # Get AUC and plot ROC
    y_true, y_score = y, decision_trees_clf.predict_proba(X)
    fp_rate, tp_rate, _ = roc_curve(y_true, y_score)
    plot_roc_curve(fp_rate, tp_rate, title='Decision Trees Receiver operating characteristic')

    return (total_loss,
            same_corr_cnt, diff_corr_cnt,
            same_incorr_cnt, diff_incorr_cnt,
            len(test_ds))


def train_svm(HP, DS, train_ds, save_wt_fname='pnet_svm_clf.pkl'):
    """
    Train the SVM

    Suggested Parameters
    batch_size=16
    """
    batch_size = HP.batch_size
    data_loader_train = DataLoader(train_ds, batch_size=16,
                                   shuffle=True)
    X, y = None, None
    # Combine the entire dataset
    for batch_idx, (_x, _y, _z, _idx, same_cls, diff_cls) in enumerate(data_loader_train):
        same_cls, diff_cls = same_cls.detach().numpy(), diff_cls.detach().numpy()
        same_target, diff_target = np.ones(
            same_cls.shape[0]), np.zeros(diff_cls.shape[0])

        if X is None and y is None:
            X = np.concatenate([same_cls, diff_cls], axis=0)
            y = np.concatenate([same_target, diff_target], axis=0)
        else:
            X = np.concatenate([X, same_cls, diff_cls], axis=0)
            y = np.concatenate([y, same_target, diff_target], axis=0)

    sgd_clf = SGDClassifier(loss='hinge', penalty='l2')

    """
    Note:
    rbf_feature reduces dimensions of training data
    but also penalizes accuracy, precision and recall of model
        rbf_feature = RBFSampler(gamma=1, random_state=1)
        X = rbf_feature.fit_transform(X)
    """
    sgd_clf.fit(X, y)
    y_pred = sgd_clf.predict(X)

    print(f"Total Hinge Loss: {hinge_loss(y, y_pred)}")
    if save_wt_fname is not None:
        _ = joblib.dump(sgd_clf, DS.CLASSIFIER_TRAINED_WEIGHT_DIR+'/'+save_wt_fname,
                        compress=9)
    return sgd_clf


def eval_svm(svm_clf, test_ds, batch_size=16):
    test_dl = DataLoader(test_ds, batch_size=batch_size,
                         shuffle=False)
    X, y = None, None
    # Combine the entire dataset
    for batch_idx, (_x, _y, _z, _idx, same_cls, diff_cls) in enumerate(test_dl):
        same_cls, diff_cls = same_cls.detach().numpy(), diff_cls.detach().numpy()
        same_target, diff_target = np.ones(
            same_cls.shape[0]), np.zeros(diff_cls.shape[0])

        if X is None and y is None:
            X = np.concatenate([same_cls, diff_cls], axis=0)
            y = np.concatenate([same_target, diff_target], axis=0)
        else:
            X = np.concatenate([X, same_cls, diff_cls], axis=0)
            y = np.concatenate([y, same_target, diff_target], axis=0)

    """
    Note:
    rbf_feature reduces dimensions of training data
    but also penalizes accuracy, precision and recall of model
        rbf_feature = RBFSampler(gamma=1, random_state=1)
        X = rbf_feature.fit_transform(X)
    """
    y_pred = svm_clf.predict(X)
    y, y_pred = y.astype(int), y_pred.astype(int)

    same_corr_cnt = np.sum(y_pred & y)
    same_incorr_cnt = np.sum(y & (y_pred ^ 1))
    diff_corr_cnt = np.sum((y ^ 1) & (y_pred ^ 1))
    diff_incorr_cnt = np.sum((y ^ 1) & y_pred)

    total_loss = hinge_loss(y, y_pred)
    print_model_metrics(total_loss, same_corr_cnt, same_incorr_cnt,
                        diff_corr_cnt, diff_incorr_cnt, len(test_ds))

    return (total_loss,
            same_corr_cnt, diff_corr_cnt,
            same_incorr_cnt, diff_incorr_cnt,
            len(test_ds))