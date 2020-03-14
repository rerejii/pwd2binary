# -*- coding: utf-8 -*-

import tensorflow as tf
from tqdm import tqdm
# from tqdm import tqdm_notebook as tqdm
import shutil
import time
import os
import numpy as np
import ShareNetFunc as nfunc
import math
import sys
import csv

class FitManager():
    def __init__(self,
                 net_cls,
                 path_cls,
                 shuf_train_ds_cls,
                 train_ds_cls,
                 test_ds_cls,
                 validation_ds_cls=None,
                 check_img_path=[], ):
        self.net_cls = net_cls
        self.path_cls = path_cls
        self.shuf_train_ds_cls = shuf_train_ds_cls
        self.train_ds_cls = train_ds_cls
        self.test_ds_cls = test_ds_cls
        self.validation_ds_cls = validation_ds_cls
        self.step_writer = tf.summary.create_file_writer(logdir=self.path_cls.get_step_log_path())
        self.check_img_path = check_img_path
        self.accuracy_csv_header = ['step', 'train_accuracy', 'test_accuracy', 'validation_accuracy',
                                    'train_small_region', 'train_middle_region', 'train_large_region',
                                    'test_small_region', 'test_middle_region', 'test_large_region',
                                    'valid_small_region', 'valid_middle_region', 'valid_large_region',]
        self.time_csv_header = ['step', 'study_time', 'total_time', 'test_time(ms)']
        self.summary_scalar_name = ['train_accuracy', 'test_accuracyc', 'validation_accuracy',
                                    'train_small_region', 'train_middle_region', 'train_large_region',
                                    'test_small_region', 'test_middle_region', 'test_large_region',
                                    'valid_small_region', 'valid_middle_region', 'valid_large_region',
                                    'test_time']
        self.summary_image_tag = ['learn', 'test', 'valid']
        self.small_region = 148
        self.middle_region = 210
        self.large_region = 256

    # ==============================================================================================================
    # ========== 外部呼出し関数 =====================================================================================
    # ==============================================================================================================
    # epoch単位のデータでシャッフルを行う関係上、step毎にデータをストックしてepoch終了時に一括で正式に保存する方式
    def fit(self,
            end_epoch,
            device_list=[0],
            ckpt_step=None,
            restore=True,):
        if ckpt_step is None:
            ckpt_step = self.shuf_train_ds_cls.get_total_data()
        if not os.path.isfile(self.path_cls.make_csv_path(filename=self.path_cls.accuracy_csv_name)):
            nfunc.write_csv(path_cls=self.path_cls, filename=self.path_cls.accuracy_csv_name, datalist=[self.accuracy_csv_header])
        if not os.path.isfile(self.path_cls.make_csv_path(filename=self.path_cls.time_csv_name)):
            nfunc.write_csv(path_cls=self.path_cls, filename=self.path_cls.time_csv_name, datalist=[self.time_csv_header])
        if restore:
            path_list = self.path_cls.search_newpoint_path(filename=self.path_cls.ckpt_epoch_name)
            if path_list:
                # tf.train.latest_checkpoint('./tf_estimator_example/')  # 別のチェックポイント指定方法
                root, ext = os.path.splitext(path_list[0])
                self.net_cls.ckpt_restore(path=root)
        total_step = self.net_cls.get_step()
        stock_eva = []
        stock_time = []
        remain_step = ckpt_step - (total_step % ckpt_step)
        epoch = self.net_cls.get_epoch()
        shuf_train_ds_iter = self.shuf_train_ds_cls.get_inited_iter()
        while self.shuf_train_ds_cls.get_remain_data() is not 0:
            if epoch >= end_epoch:
                data = csv.reader(self.path_cls.epoch_accuracy_csv_name)
                epoch_log_writer = tf.summary.create_file_writer(logdir=self.path_cls.get_epoch_log_path())
                nfunc.accuracy_log_write(epoch_log_writer, data, self.accuracy_csv_header)
                print('finish!')
                break # fit関数終了
            # ===== 学習 =====
            start_time = time.time()
            pbar = tqdm(total=self.shuf_train_ds_cls.get_total_data(), desc='fitting (epoch '+str(epoch+1)+')', leave=False)
            pbar.update(self.shuf_train_ds_cls.get_total_data()-self.shuf_train_ds_cls.get_remain_data())
            while remain_step > 0 and self.shuf_train_ds_cls.get_remain_data() != 0:
                # データ残数によってマルチデバイスかシングルデバイスかの判定を行う
                need_data_n = (self.shuf_train_ds_cls.get_batch_size() * len(device_list))
                if self.shuf_train_ds_cls.get_remain_data() >= need_data_n and remain_step >= need_data_n:
                    # マルチデバイス
                    self.net_cls.multi_train_step(ds_iter=shuf_train_ds_iter, device_list=device_list)
                    self.shuf_train_ds_cls.data_used_apply(use_count=len(device_list))
                    progress_step = len(device_list)*self.shuf_train_ds_cls.get_batch_size()
                else:
                    # シングルデバイス
                    progress_step = self.shuf_train_ds_cls.get_next_data_n()
                    self.net_cls.train_step(ds_iter=shuf_train_ds_iter, device_list=device_list, rate=self.shuf_train_ds_cls.get_next_data_rate())
                    self.shuf_train_ds_cls.data_used_apply(use_count=1)
                remain_step -= progress_step
                total_step += progress_step
                pbar.update(progress_step)  # プロセスバーを進行
            pbar.close()  # プロセスバーの終了
            self.net_cls.add_study_time(time.time() - start_time)
            # ===== 評価 =====
            train_accuracy, train_region_acc = self.check_accuracy(
                ds_cls=self.train_ds_cls,
                device_list=device_list,
                task_name='train_ds check (epoch:'+str(epoch+1)+' step:'+str(total_step)+')'
            )
            test_start_time = time.time()
            test_accuracy, test_region_acc = self.check_accuracy(
                ds_cls=self.test_ds_cls,
                device_list=device_list,
                task_name='test_ds check (epoch:'+str(epoch+1)+' step:'+str(total_step)+')'
            )
            test_time = time.time() - test_start_time
            if self.validation_ds_cls is not None:
                validation_accuracy, validation_region_acc = self.check_accuracy(
                                          ds_cls=self.validation_ds_cls,
                                          device_list=device_list,
                                          task_name='validation_ds check (epoch:'+str(epoch+1)+' step:'+str(total_step)+')'
                                          )
            else:
                validation_accuracy = test_accuracy
                validation_region_acc = [0.0, 0.0, 0.0]
            print(str(epoch+1)+'epoch '+str(total_step)+'step accuracy [train, test, valid]: ['
                  +str(train_accuracy)+', '+str(test_accuracy)+', '+str(validation_accuracy)+']')
            print(str(epoch + 1) + 'epoch ' + str(total_step) + 'test region [small, middle, large]: ['
                  + str(test_region_acc[0]) + ', ' + str(test_region_acc[1]) + ', ' + str(test_region_acc[2]) + ']')
            self.net_cls.add_total_time(time.time() - start_time)
            # ===== 保存 =====
            study_time_str = self.net_cls.get_str_study_time()
            total_time_str = self.net_cls.get_str_total_time()
            test_time_s = math.floor(test_time)
            test_time_s_norm = math.floor(test_time + 54000)
            latest_eva = [total_step, train_accuracy, test_accuracy, validation_accuracy,
                          train_region_acc[0], train_region_acc[1], train_region_acc[2],
                          test_region_acc[0], test_region_acc[1], test_region_acc[2],
                          validation_region_acc[0], validation_region_acc[1], validation_region_acc[2],
                          test_time_s]
            latest_time = [total_step, study_time_str, total_time_str, test_time_s_norm]
            if remain_step <= 0:
                self._step_proc(total_step, latest_eva)
                stock_eva.append(latest_eva)
                stock_time.append(latest_time)
                remain_step = remain_step + ckpt_step
            if self.shuf_train_ds_cls.get_remain_data() == 0:
                print('saveing now!')
                epoch += 1
                self._epoch_proc(epoch, total_step, stock_eva, stock_time, latest_eva, latest_time)
                stock_eva = []
                stock_time = []
                shuf_train_ds_iter = self.shuf_train_ds_cls.get_inited_iter()

    # ==============================================================================================================
    # ========== check_accuracy関数 ================================================================================
    # ==============================================================================================================
    def check_accuracy(self, ds_cls, device_list, task_name='check_accuracy'):
        total_accuracy = 0
        total_region_acc = np.array([0., 0., 0.,])
        ds_iter = ds_cls.get_inited_iter()
        pbar = tqdm(total=ds_cls.get_remain_data(), desc=task_name, leave=False)  # プロセスバーの設定
        while ds_cls.get_remain_data() is not 0:
            if ds_cls.get_remain_data() >= (ds_cls.get_batch_size() * len(device_list)):
                # マルチデバイス
                accuracy, region_acc = self.multi_check_step(ds_iter=ds_iter, device_list=device_list, data_n=ds_cls.get_next_data_n())
                ds_cls.data_used_apply(use_count=len(device_list))
                progress_step = len(device_list)*ds_cls.get_batch_size()
            else:
                # シングルデバイス
                accuracy, region_acc = self.check_step(ds_iter=ds_iter, gpu_index=device_list[0], data_n=ds_cls.get_next_data_n())
                ds_cls.data_used_apply(use_count=1)
                progress_step = ds_cls.get_next_data_n()
            total_accuracy += accuracy
            total_region_acc += np.array(region_acc)
            pbar.update(progress_step)  # プロセスバーを進行
        ave_accuracy = total_accuracy / ds_cls.get_total_data()
        total_region_acc = total_region_acc / ds_cls.get_total_data()
        pbar.close()  # プロセスバーの終了
        return ave_accuracy.numpy(), total_region_acc

    @tf.function  # 関数をグラフモードで実行 https://www.tensorflow.org/guide/autograph?hl=ja
    def multi_check_step(self, ds_iter, device_list, data_n):
        accuracy_list = []
        small_acc_list = []
        middle_acc_list = []
        large_acc_list = []
        for gpu_index in device_list:
            with tf.device('/gpu:%d' % gpu_index):  # gpu単位の処理
                input_image, target = next(ds_iter)
                target = nfunc.target_cut_padding(target=target, padding=self.net_cls.get_padding())
                # Generatorによる画像生成
                generator = self.net_cls.get_generator()
                gen_output = generator(input_image, training=True)
                acc, region_acc_list = nfunc.evaluate(net_cls=self.net_cls, out=gen_output, ans=target,
                                                      region=[self.small_region, self.middle_region, self.large_region],
                                                      batch_data_n=int(data_n/len(device_list)), )
                accuracy_list.append(acc * data_n)
                small_acc_list.append(region_acc_list[0] * data_n)
                middle_acc_list.append(region_acc_list[1] * data_n)
                large_acc_list.append(region_acc_list[2] * data_n)
        return sum(accuracy_list), [sum(small_acc_list), sum(middle_acc_list), sum(large_acc_list)]

    # self.small_region = 148
    # self.middle_region = 210
    # self.large_region = 256

    @tf.function  # 関数をグラフモードで実行 https://www.tensorflow.org/guide/autograph?hl=ja
    def check_step(self, ds_iter, gpu_index, data_n):
        accuracy_list = []
        with tf.device('/gpu:%d' % gpu_index):  # gpu単位の処理
            input_image, target = next(ds_iter)
            target = nfunc.target_cut_padding(target=target, padding=self.net_cls.get_padding())
            # Generatorによる画像生成
            generator = self.net_cls.get_generator()
            gen_output = generator(input_image, training=True)
            acc, region_acc_list = nfunc.evaluate(net_cls=self.net_cls, out=gen_output, ans=target,
                                                  region=[self.small_region, self.middle_region, self.large_region],
                                                  batch_data_n=int(data_n), )
            acc = (acc * data_n)
            small_acc = (region_acc_list[0] * data_n)
            middle_acc = (region_acc_list[1] * data_n)
            large_region = (region_acc_list[2] * data_n)
        return acc, [small_acc, middle_acc, large_region]

    # ==============================================================================================================
    # ========== 保存関数 ==========================================================================================
    # ==============================================================================================================
    def _step_proc(self, total_step, latest_eva):
        # ckptのstep値更新
        self.net_cls.set_ckpt_val(step_val=total_step)
        # stock
        stock_path = self.path_cls.make_stock_path(filename=self.path_cls.ckpt_step_name+str(total_step))
        self.net_cls.get_checkpoint().save(stock_path)
        # best_accuracy更新
        check_t = self.net_cls.update_check_best_test_accuracy(latest_eva[2])
        check_v = self.net_cls.update_check_best_validation_accuracy(latest_eva[3])
        if check_t:
            old_path = self.path_cls.search_best_path(filename=self.path_cls.ckpt_step_name)
            best_path = self.path_cls.make_best_path(filename=self.path_cls.ckpt_step_name+str(total_step))
            self.net_cls.get_checkpoint().save(best_path)
            for path in old_path:
                os.remove(path)
        # 画像の保存
        nfunc.img_check(step=total_step, net_cls=self.net_cls,
                        path_cls=self.path_cls, check_img_path=self.check_img_path)

    def _epoch_proc(self, epoch, total_step, stock_eva, stock_time, latest_eva, latest_time):
        latest_eva_li = [latest_eva]
        latest_time_li = [latest_time]
        # save
        self.net_cls.set_ckpt_val(step_val=total_step, epoch_val=epoch)
        for stock_ckpt in self.path_cls.search_stock_path(filename=self.path_cls.ckpt_step_name):
            shutil.move(stock_ckpt, self.path_cls.make_checkpoint_path(filename=''))
        # epoch単位のckpt(復元用)
        old_path = self.path_cls.search_newpoint_path(filename=self.path_cls.ckpt_epoch_name)
        epoch_path = self.path_cls.make_newpoint_path(filename=self.path_cls.ckpt_epoch_name+str(epoch))
        self.net_cls.get_checkpoint().save(epoch_path)
        for path in old_path:
            os.remove(path)
        # ログの書き込み
        # step_writer = tf.summary.create_file_writer(logdir=self.path_cls.get_step_log_path())
        nfunc.log_write(writer=self.step_writer, stock=stock_eva,
                        scalar_name=self.summary_scalar_name, image_tag=self.summary_image_tag,
                        check_img_path=self.check_img_path, path_cls=self.path_cls)
        nfunc.write_csv(path_cls=self.path_cls, filename=self.path_cls.accuracy_csv_name, datalist=stock_eva)
        nfunc.write_csv(path_cls=self.path_cls, filename=self.path_cls.time_csv_name, datalist=stock_time)
        nfunc.write_csv(path_cls=self.path_cls, filename=self.path_cls.epoch_accuracy_csv_name, datalist=latest_eva_li)
        nfunc.write_csv(path_cls=self.path_cls, filename=self.path_cls.epoch_time_csv_name, datalist=latest_time_li)

