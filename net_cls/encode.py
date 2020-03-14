# -*- coding: utf-8 -*-

import tensorflow as tf
import time
import math
import ShareNetFunc as nfunc


class NetManager:
    def __init__(self,
                 loss_object=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                 gen_optimizer=tf.keras.optimizers.Adam(0.01),
                 output_channel=1,
                 lambda_val=100,
                 ):
        self.output_channel = output_channel
        self.gen = self._Generator()
        self.loss_object = loss_object
        self.gen_optimizer = gen_optimizer
        self.lambda_val = lambda_val
        self.best_test_accuracy = tf.Variable(initial_value=0.0,
                                              trainable=False,
                                              dtype=tf.float32,
                                              name='best_test_accuracy')
        self.best_validation_accuracy = tf.Variable(initial_value=0.0,
                                                    trainable=False,
                                                    dtype=tf.float32,
                                                    name='best_validation_accuracy')
        self.step = tf.Variable(initial_value=0, trainable=False, dtype=tf.int32, name='step')
        self.epoch = tf.Variable(initial_value=0, trainable=False, dtype=tf.int32, name='epoch')
        self.study_time = tf.Variable(initial_value=0.0, trainable=False, dtype=tf.float32, name='study_time')
        self.total_time = tf.Variable(initial_value=0.0, trainable=False, dtype=tf.float32, name='total_time')
        self.checkpoint = tf.train.Checkpoint(
            generator_optimizer=self.gen_optimizer,
            generator=self.gen,
            best_validation_accuracy=self.best_validation_accuracy,
            best_test_accuracy=self.best_test_accuracy,
            step=self.step,
            epoch=self.epoch,
            study_time=self.study_time,
            total_time=self.total_time,
            )
        self.time_basis = 54000  # timeの基準
        self.day_time = 86400  # 1日
        self.padding = 28

# ************ クラス外呼び出し関数 ************

    # ===== ckpt関係 =====
    def ckpt_restore(self, path):
        self.checkpoint.restore(path)

    def get_str_study_time(self):
        ms_time, s_time = math.modf(self.study_time.numpy() + self.time_basis)  # ミニセカンド セカンド
        day, times = divmod(s_time, self.day_time)  # 日数と時間に
        day = int(day)
        step_times = time.strptime(time.ctime(times))
        str_time = str(day) + ':' + time.strftime("%H:%M:%S", step_times) + str(ms_time)[1:]
        return str_time

    def get_str_total_time(self):
        ms_time, s_time = math.modf(self.total_time.numpy() + self.time_basis)  # ミニセカンド セカンド
        day, times = divmod(s_time, self.day_time)  # 日数と時間に
        day = int(day)
        step_times = time.strptime(time.ctime(times))
        str_time = str(day) + ':' + time.strftime("%H:%M:%S", step_times) + str(ms_time)[1:]
        return str_time

    def get_epoch(self):
        return self.epoch.numpy()

    def get_step(self):
        return self.step.numpy()

    def add_study_time(self, proc_time):
        self.study_time.assign(self.study_time + proc_time)

    def add_total_time(self, proc_time):
        self.total_time.assign(self.total_time + proc_time)

    def update_check_best_validation_accuracy(self, accuracy):
        if accuracy > self.best_validation_accuracy:
            self.best_validation_accuracy = accuracy
            return True
        return False

    def update_check_best_test_accuracy(self, accuracy):
        if accuracy > self.best_test_accuracy:
            self.best_test_accuracy = accuracy
            return True
        return False

    def get_checkpoint(self):
        return self.checkpoint

    # ===== ネットワーク関係 =====
    def get_padding(self):
        return self.padding

    def get_generator(self):
        return self.gen

    def set_ckpt_val(self, step_val=None, epoch_val=None):
        if step_val is not None:
            self.step.assign(step_val)
        if epoch_val is not None:
            self.epoch.assign(epoch_val)

    def get_generator_optimizer(self):
        return self.gen_optimizer

    def generator_loss(self, gen_output, target):  # 値の範囲を確認
        gen_loss = self.loss_object(target, gen_output)
        return gen_loss

    # ===== 値変換系 =====
    def binary_from_img(self, data):
        return tf.greater_equal(data, 255)

    def binary_from_data(self, data):
        return tf.greater_equal(data, 0.5)

    def img_from_netdata(self, data):
        return data * 255

    def netdata_from_img(self, data):
        return data / 255

    # ==============================================================================================================
    # ========== train関数 =========================================================================================
    # ==============================================================================================================
    @tf.function  # 関数をグラフモードで実行 https://www.tensorflow.org/guide/autograph?hl=ja
    def multi_train_step(self, ds_iter, device_list):
        generator_gradients_list = []
        for gpu_index in device_list:
            with tf.device('/gpu:%d' % gpu_index):  # gpu単位の処理
                # tf.GradientTape()以下に勾配算出対象の計算を行う https://qiita.com/propella/items/5b2182b3d6a13d20fefd
                with tf.GradientTape() as gen_tape:
                    input_image, target = next(ds_iter)
                    target = nfunc.target_cut_padding(target=target, padding=self.get_padding())
                    # Generatorによる画像生成
                    generator = self.get_generator()
                    gen_output = generator(input_image, training=True)
                    # loss算出
                    gen_loss = self.generator_loss(gen_output, target)
                # 勾配算出 trainable_variables:訓練可能な変数
                generator_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
                # 後で平均を取る為に保存
                generator_gradients_list.append(generator_gradients)
                # gpu単位の処理ここまで
        # with tf.device('/gpu:%d' % device_list[0]):  # gpu単位の処理
        generator = self.get_generator()
        # 勾配の平均、怪しい
        generator_gradients_average = nfunc.average_gradients(generator_gradients_list)
        # 勾配の適用
        self.get_generator_optimizer().apply_gradients(
            zip(generator_gradients_average, generator.trainable_variables))
        return

    @tf.function  # 関数をグラフモードで実行 https://www.tensorflow.org/guide/autograph?hl=ja
    def train_step(self, ds_iter, device_list, rate=1):
        with tf.device('/gpu:%d' % device_list[0]):  # gpu単位の処理
            # tf.GradientTape()以下に勾配算出対象の計算を行う https://qiita.com/propella/items/5b2182b3d6a13d20fefd
            with tf.GradientTape() as gen_tape:
                input_image, target = next(ds_iter)
                target = nfunc.target_cut_padding(target=target, padding=self.get_padding())
                # Generatorによる画像生成
                generator = self.get_generator()
                gen_output = generator(input_image, training=True)
                # loss算出
                gen_loss = self.generator_loss(gen_output, target)
            # 勾配算出 trainable_variables:訓練可能な変数
            generator_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
            # バッチサイズ毎のデータサイズの割合を勾配に適用させる 1/len(device_list)を掛けて、複数GPU時の時との差を調整
            generator_gradients = nfunc.rate_multiply(generator_gradients, rate * (1 / len(device_list)))
            # gpu単位の処理ここまで
        # with tf.device('/gpu:%d' % gpu_index):  # gpu単位の処理
        generator = self.get_generator()
        # 勾配の適用
        self.get_generator_optimizer().apply_gradients(
            zip(generator_gradients, generator.trainable_variables))
        return

    # ==============================================================================================================
    # ========== check_accuracy関数 ================================================================================
    # ==============================================================================================================
    @tf.function  # 関数をグラフモードで実行 https://www.tensorflow.org/guide/autograph?hl=ja
    def multi_check_step(self, ds_iter, device_list, data_n):
        accuracy_list = []
        for gpu_index in device_list:
            with tf.device('/gpu:%d' % gpu_index):  # gpu単位の処理
                input_image, target = next(ds_iter)
                target = nfunc.target_cut_padding(target=target, padding=self.get_padding())
                # Generatorによる画像生成
                generator = self.get_generator()
                gen_output = generator(input_image, training=True)
                accuracy_list.append(nfunc.evaluate(net_cls=self, out=gen_output, ans=target) * data_n)
        return sum(accuracy_list)

    @tf.function  # 関数をグラフモードで実行 https://www.tensorflow.org/guide/autograph?hl=ja
    def check_step(self, ds_iter, gpu_index, data_n):
        accuracy_list = []
        with tf.device('/gpu:%d' % gpu_index):  # gpu単位の処理
            input_image, target = next(ds_iter)
            target = nfunc.target_cut_padding(target=target, padding=self.get_padding())
            # Generatorによる画像生成
            generator = self.get_generator()
            gen_output = generator(input_image, training=True)
            accuracy_list.append(nfunc.evaluate(net_cls=self, out=gen_output, ans=target) * data_n)
        return sum(accuracy_list)

# ************ クラス内呼び出し関数 ************
    # ==============================================================================================================
    # ========== NETWORK ================================================================================
    # ==============================================================================================================
    def _Generator(self):
        down_stack = [
            self._relu_conv_valid_layer(40, 3),  # (bs, 256, 256, 40) 1
            self._downsample(80, 3),  # (bs, 128, 128, 80) 2
        ]
        cnn_stack = [
            self._relu_conv_valid_layer(80, 3),  # (bs, 128, 128, 80) 3
            self._relu_conv_valid_layer(80, 3),  # (bs, 128, 128, 80) 4
            self._relu_conv_valid_layer(80, 3),  # (bs, 128, 128, 80) 5
            self._relu_conv_valid_layer(80, 3),  # (bs, 128, 128, 80) 6
            self._relu_conv_valid_layer(80, 3),  # (bs, 128, 128, 80) 7
            self._relu_conv_valid_layer(80, 3),  # (bs, 128, 128, 80) 8
            self._relu_conv_valid_layer(80, 3),  # (bs, 128, 128, 80) 9
            self._relu_conv_valid_layer(80, 3),  # (bs, 128, 128, 80) 10
            self._relu_conv_valid_layer(80, 3),  # (bs, 128, 128, 80) 11
            self._relu_conv_valid_layer(80, 3),  # (bs, 128, 128, 80) 12
            self._relu_conv_valid_layer(80, 3),  # (bs, 128, 128, 80) 13
            self._relu_conv_valid_layer(80, 3),  # (bs, 128, 128, 80) 14
        ]
        initializer = tf.random_normal_initializer(0., 0.01)
        last = tf.keras.layers.Conv2D(filters=self.output_channel,
                                      kernel_size=3,
                                      strides=1,
                                      padding='valid',
                                      kernel_initializer=initializer,
                                      use_bias=True,
                                      bias_initializer=initializer,
                                      activation='sigmoid')
        inputs = tf.keras.layers.Input(shape=[None, None, 3])
        x = inputs
        for down in down_stack:
            x = down(x)
        for cnn in cnn_stack:
            x = cnn(x)
        x = self._upsample_backend(x, 40, 3)  # 15
        x = last(x)  # 16
        return tf.keras.Model(inputs=inputs, outputs=x)



    # backends https://keras.io/ja/backend/
    def _upsample_backend(self, x, filters, size, apply_batchnorm=True):
        x = tf.keras.backend.resize_images(x,
                                           height_factor=2,
                                           width_factor=2,
                                           data_format='channels_last',
                                           interpolation='nearest',
                                           )
        initializer = tf.random_normal_initializer(0., 0.02)
        sequential = tf.keras.Sequential()
        sequential.add(
            tf.keras.layers.Conv2D(filters, size, strides=1, padding='valid',
                                   kernel_initializer=initializer, use_bias=True,
                                   bias_initializer=initializer,)
        )
        if apply_batchnorm:
            sequential.add(tf.keras.layers.BatchNormalization())
        sequential.add(tf.keras.layers.ReLU())
        return sequential(x)


    def _relu_conv_valid_layer(self, filters, size, apply_batchnorm=True):
        initializer = tf.random_normal_initializer(0., 0.02)
        result = tf.keras.Sequential()
        result.add(
            tf.keras.layers.Conv2D(filters, size, strides=1, padding='valid',
                                   kernel_initializer=initializer, use_bias=True,
                                   bias_initializer=initializer,)
        )
        if apply_batchnorm:
            result.add(tf.keras.layers.BatchNormalization())
        result.add(tf.keras.layers.ReLU())
        return result

    def _downsample(self, filters, size, apply_batchnorm=True):
        initializer = tf.random_normal_initializer(0., 0.02)
        result = tf.keras.Sequential()
        result.add(
            tf.keras.layers.Conv2D(filters, size, strides=2, padding='valid',
                                   kernel_initializer=initializer, use_bias=True,
                                   bias_initializer=initializer,)
        )
        if apply_batchnorm:
            result.add(tf.keras.layers.BatchNormalization())
        result.add(tf.keras.layers.ReLU())
        return result
