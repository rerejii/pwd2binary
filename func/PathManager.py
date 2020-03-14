# -*- coding: utf-8 -*-

import tensorflow as tf
import sys
import glob
import os
from tqdm import tqdm

class PathManager():
    def __init__(self, tfrecord_folder, output_rootfolder='Output_set', epoch_output_rootfolder=None):
        # tf_recordが存在するフォルダ
        self.tfrecord_folder = tfrecord_folder
        # self.out_folder_name = out_folder_name
        self.train_ds_path = glob.glob(self.tfrecord_folder + '/' + 'LearnData*.tfrecords')[0]
        self.test_ds_path = glob.glob(self.tfrecord_folder + '/' + 'TestData*.tfrecords')[0]
        # self.validation_ds_path = glob.glob(self.tfrecord_folder + '/' + 'ValidationData*.tfrecords')
        valid_glob = glob.glob(self.tfrecord_folder + '/' + 'ValidationData*.tfrecords')
        if valid_glob:
            self.validation_ds_path = valid_glob[0]
        else:
            self.validation_ds_path = None
        self.property_path = glob.glob(self.tfrecord_folder + '/' + 'Property*.csv')[0]
        # 基準とするフォルダ
        self.output_rootfolder = output_rootfolder
        # 学習途中経過のモデルデータを保存するフォルダ
        self.checkpoint_folder = self.output_rootfolder + '/' + 'CheckpointModel'
        # 学習途中経過のモデルデータを保存するフォルダ
        self.newpoint_folder = self.output_rootfolder + '/' + 'NewpointModel'
        # epochが終了するまでの物を保存するフォルダ
        self.stock_model_folder = self.output_rootfolder + '/' + 'StockModel'
        # 画像を保存するフォルダ
        self.step_image_folder = self.output_rootfolder + '/' + 'StepImage'
        # ベストのモデルデータを保存するフォルダ
        self.best_folder = self.output_rootfolder + '/' + 'BestModel'
        # TensorBoard用のデータを書き出すフォルダ
        self.step_log_folder = self.output_rootfolder + '/' + 'StepLog'
        # csvファイル用
        self.csv_folder = self.output_rootfolder + '/' + 'CsvDatas'
        self.epoch_output_rootfolder = epoch_output_rootfolder \
            if epoch_output_rootfolder is not None else output_rootfolder
        self.epoch_log_folder = self.epoch_output_rootfolder + '/' + 'EpochLog'

        self.accuracy_csv_name = 'step_accuracy.csv'
        self.time_csv_name = 'step_time.csv'
        self.epoch_accuracy_csv_name = 'epoch_accuracy.csv'
        self.epoch_time_csv_name = 'epoch_time.csv'
        self.ckpt_step_name = 'ckpt-step'
        self.ckpt_epoch_name = 'ckpt-epoch'
        # self.ckpt_name_step = 'ckpt_step'
        # self.ckpt_name_epoch = 'ckpt_epoch'
        # self.ckpt_name_best = 'ckpt_best'


    def all_makedirs(self):
        os.makedirs(self.checkpoint_folder, exist_ok=True)
        os.makedirs(self.best_folder, exist_ok=True)
        os.makedirs(self.newpoint_folder, exist_ok=True)
        os.makedirs(self.csv_folder, exist_ok=True)
        os.makedirs(self.stock_model_folder, exist_ok=True)
        os.makedirs(self.step_image_folder, exist_ok=True)
        os.makedirs(self.step_log_folder, exist_ok=True)
        os.makedirs(self.epoch_log_folder, exist_ok=True)


    def get_property_path(self):
        return self.property_path


    def get_train_ds_path(self):
        return self.train_ds_path


    def get_test_ds_path(self):
        return self.test_ds_path


    def get_validation_ds_path(self):
        return self.validation_ds_path


    def get_image_folder_path(self):
        return self.step_image_folder


    def get_epoch_log_path(self):
        return self.epoch_log_folder


    def get_step_log_path(self):
        return self.step_log_folder


    def make_checkpoint_path(self, filename):
        return self.checkpoint_folder + '/' + filename


    def make_newpoint_path(self, filename):
        return self.newpoint_folder + '/' + filename


    def make_stock_path(self, filename):
        return self.stock_model_folder + '/' + filename


    def make_best_path(self, filename):
        return self.best_folder + '/' + filename


    def make_epoch_log_path(self, filename):
        return self.epoch_log_folder + '/' + filename


    def make_step_log_path(self, filename):
        return self.best_folder + '/' + filename


    def make_csv_path(self, filename):
        return self.csv_folder + '/' + filename


    def make_model_path(self, filename):
        return self.output_rootfolder + '/' + filename


    def search_newpoint_path(self, filename):
        return glob.glob(self.newpoint_folder + '/' + filename + '*')


    def search_stock_path(self, filename):
        return glob.glob(self.stock_model_folder + '/' + filename + '*')


    def search_best_path(self, filename):
        return glob.glob(self.best_folder + '/' + filename + '*')
