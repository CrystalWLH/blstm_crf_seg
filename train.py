# coding=utf-8
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim

import argparse
import datetime
import pickle
import time
import os
import sys

from model.model import *
from utils.utils import *
import utils.dataLoader as dataLoader
import utils.dataPreprocess as dataPreprocess

try:
    import tensorflow as tf
except ImportError:
    print("Tensorflow not installed; No tensorboard logging.")
    tf = None

def add_summary_value(writer, key, value, iteration):
    if writer is None:
        return
    summary = tf.Summary(value=[tf.Summary.Value(tag=key, simple_value=value)])
    writer.add_summary(summary, iteration)

def get_lr(optimizer):
    for group in optimizer.param_groups:
        return group['lr']

class Logger(object):
    def __init__(self, file_name):
        self.terminal = sys.stdout
        self.log = open(file_name, 'a')
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        self.terminal.flush()
        self.log.flush()

class BiLSTM_CRF(nn.Module):
    '''
    几个模块的拼接，测试
    '''
    def __init__(self, opt):
        super(BiLSTM_CRF, self).__init__()
        self.opt = opt
        self.embeds = WordEmbedding(self.opt)
        self.opt.tagset_size += 2    # for [START_TAG] and [STOP_TAG]
        self.bilstm = BiLSTM(self.opt)
        opt.tagset_size -= 2    # done inside CRF
        self.CRF = CRF(self.opt)

    def forward(self, packed_sent):
        packed_embeds = self.embeds(packed_sent)
        packed_feats = self.bilstm(packed_embeds)
        score, best_paths = self.CRF.decode(packed_feats)
        return score, best_paths

    def neg_log_likelihood(self, packed_sent, packed_tags):
        packed_embeds = self.embeds(packed_sent)
        packed_feats = self.bilstm(packed_embeds)
        score, best_paths = self.CRF.decode(packed_feats)
        return self.CRF.neg_log_likelihood(packed_feats, packed_tags), score, best_paths

def train(model, optimizer, dataloader, fileTemp):
    model.train()
    stime = time.time()
    iterloss = 0.0
    totalLoss = 0.0
    totalCount = 0.0
    iter_cnt = 0
    for (packed_sent, packed_tag), idx_unsort, words in dataloader:
        model.zero_grad()

        loss, score, best_paths = model.neg_log_likelihood(packed_sent, packed_tag)
        lastloss = loss.item()

        iterloss = iterloss + lastloss
        iter_cnt = iter_cnt + 1

        totalLoss = totalLoss + lastloss
        totalCount = totalCount + 1

        loss.backward()
        optimizer.step()

        if iter_cnt != 0 and iter_cnt % 100 == 0:
            print("Step: %d \t loss: %f" %(iter_cnt, iterloss / 100.0))
            fileTemp.flush()
            iterloss = 0.0

    return totalLoss / totalCount


def evaluate(model, validdataloader):
    model.eval()
    totalLoss = 0.0
    precission = 0.0
    recall = 0.0
    f1_measure = 0.0
    count = 0.0
    for (packed_sent, packed_tag), idx_unsort, words in validdataloader:
        count = count + 1
        with torch.no_grad():
            loss, score, best_paths = model.neg_log_likelihood(packed_sent, packed_tag)
            p, r, f = calculatePRF(best_paths, packed_tag)
        totalLoss = totalLoss + loss.item()
        precission = precission + p
        recall = recall + r
        f1_measure = f1_measure + f

    return totalLoss / count, precission / count, recall / count, f1_measure / count
    


if __name__ == "__main__":

    ckpt_path = "./checkpoint/"
    data_path = "./data/"
    log_dir = "./log/"
    epochs = 40

    time_suffix = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    log_dir = log_dir + "run_%s/" % (time_suffix)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    stime = time.time()
    sys.stdout = Logger(log_dir + "log")
    tf_summary_writer = tf and tf.summary.FileWriter(log_dir + "tflog")
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\nDevice: %s" % (device))

    with open(data_path + 'vocab_tag.pkl', 'rb') as f:
        word_to_ix, ix_to_word, tag_to_ix, ix_to_tag = pickle.load(f)
    opt = argparse.Namespace()
    opt.device = device
    opt.corpus = data_path + 'train_corpus.pkl'
    opt.vocab_tag = data_path + 'vocab_tag.pkl'
    opt.embedding_dim = 64
    opt.hidden_dim = 128
    opt.batch_size = 5
    opt.vocab_size = len(word_to_ix)
    opt.tagset_size = len(tag_to_ix)

    opt.lr = 1e-4
    opt.weight_decay = 1e-4
    opt.epoch = 0       # if non-zero, load checkpoint at iter (#iter_cnt)

    # Load data (Train, Valid, Test)
    dataset = dataLoader.DataSet(opt)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=opt.batch_size, collate_fn=dataLoader.collate, drop_last=True, shuffle=True)

    opt.corpus = data_path + 'valid_corpus.pkl'

    validdataset = dataLoader.DataSet(opt)
    validdataloader = torch.utils.data.DataLoader(
        validdataset, batch_size=opt.batch_size, collate_fn=dataLoader.collate, drop_last=True)

    opt.corpus = data_path + 'test_corpus.pkl'

    testdataset = dataLoader.DataSet(opt)
    testdataloader = torch.utils.data.DataLoader(
        testdataset, batch_size=opt.batch_size, collate_fn=dataLoader.collate, drop_last=True)

    print("All necessites prepared, time used: %f s\n" % (time.time() - stime))

    # Build model
    model = BiLSTM_CRF(opt).to(device)

    if opt.epoch > 0:
        try:
            print("Load checkpoint at %s" %(ckpt_path + "base_e64_h128_iter%d.cpkt" % (opt.epoch)))
            # load parameters from checkpoint given
            model.load_state_dict(torch.load(ckpt_path + "base_e64_h128_iter%d.cpkt" % (opt.epoch)))
            print("Success\n")
        except Exception as e:
            print("Failed, check the path and permission of the checkpoint")
            exit(0)

    optimizer = optim.SGD(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)

    # Check predictions before training
    with torch.no_grad():
        for packed_sent, idx_unsort, words in testdataloader:
            _, packed_tag = model(packed_sent)
            visualize(packed_sent, packed_tag, ix_to_word, ix_to_tag, idx_unsort, words)
            break

    epoch_pre = opt.epoch

    best_loss = None
    best_model_path = None
    bad_count = 0
    
    for epoch in range(epoch_pre + 1, epochs):  
        fileTemp = sys.stdout
        train_loss = train(model, optimizer, dataloader, fileTemp)
        print("Epoch: %d \t Train_loss: %f" %(epoch, train_loss))
        sys.stdout.flush()
        
        val_loss, val_precision, val_recall, val_f1_measure = evaluate(model, validdataloader)
        print("Epoch: %d \t Valid_loss: %f \t Valid precision: %f \t Valid Recall: %f \t Valid F1_measure: %f" %(epoch, val_loss, val_precision, val_recall, val_f1_measure))
        sys.stdout.flush()

        if best_loss == None or best_loss > val_loss:
            best_loss = val_loss
            best_model_path = ckpt_path + "base_e64_h128_iter%d.cpkt" % (epoch)
            try:
                torch.save(model.state_dict(), ckpt_path + "base_e64_h128_iter%d.cpkt" % (epoch))
                print("checkpoint saved at \'%s\'" % (ckpt_path + "base_e64_h128_iter%d.cpkt" % (epoch)))
            except Exception as e:
                print(e)
        else:
            bad_count = bad_count + 1
            if bad_count == 5:
                print("Early stopping! Valid loss has increased 5 times.")
                sys.stdout.flush()
                break

    print("--------------------Testing----------------")
    sys.stdout.flush()
    print("Load the best model " + best_model_path)
    sys.stdout.flush()
    model.load_state_dict(torch.load(best_model_path))
    test_loss, test_precision, test_recall, test_f1_measure = evaluate(model, testdataloader)
    print("Test_loss: %f \t Test precision: %f \t Test Recall: %f \t Test F1_measure: %f" %(test_loss, test_precision, test_recall, test_f1_measure))
    sys.stdout.flush()
