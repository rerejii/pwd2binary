# -*- coding: utf-8 -*-

import glob
import tensorflow as tf
import time
import math
import ShareNetFunc as nfunc

# - Discriminator_Loss
#     - 弁別子損失関数は2つの入力を取ります。実画像、生成画像
#     - real_lossは、実画像と1つの配列のシグモイドクロスエントロピー損失です（これらは実画像であるため）
#     - generated_lossは、生成された画像とゼロの配列のシグモイドクロスエントロピー損失です（これらは偽の画像であるため）
#     - 次に、total_lossはreal_lossとgenerated_lossの合計です
# - Generator_Loss
#     - これは、生成された画像と1の配列のシグモイドクロスエントロピー損失です。
#     - 紙はまた、生成された画像とターゲット画像とのMAEであるL1損失を（絶対平均誤差）を含みます。
#     - これにより、生成された画像が構造的にターゲット画像に似たものになります。
#     - 総発電機損失= gan_loss + LAMBDA * l1_lossを計算する式。ここでLAMBDA = 100です。この値は、論文の著者によって決定されました。


class NetManager:
    def __init__(self,
                 loss_object=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                 gen_optimizer=tf.keras.optimizers.Adam(0.0001),
                 # dis_optimizer=tf.keras.optimizers.Adam(2e-4, beta_1=0.5),
                 output_channel=1,
                 lambda_val=100,):
        self.output_channel = output_channel
        self.gen = self._Generator()
        self.dis = self._Discriminator()
        self.loss_object = loss_object
        self.gen_optimizer = gen_optimizer
        # self.dis_optimizer = dis_optimizer
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
            # discriminator_optimizer=self.dis_optimizer,
            generator=self.gen,
            discriminator=self.dis,
            best_validation_accuracy=self.best_validation_accuracy,
            best_test_accuracy=self.best_test_accuracy,
            step=self.step,
            epoch=self.epoch,
            study_time=self.study_time,
            total_time=self.total_time,
            )
        self.time_basis = 54000  # timeの基準
        self.day_time = 86400  # 1日
        self.padding = 0

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

    def get_discriminator(self):
        return self.dis

    def set_ckpt_val(self, step_val=None, epoch_val=None):
        if step_val is not None:
            self.step.assign(step_val)
        if epoch_val is not None:
            self.epoch.assign(epoch_val)

    def get_generator_optimizer(self):
        return self.gen_optimizer

    # def get_discriminator_optimizer(self):
    #     return self.dis_optimizer

    def generator_loss(self, gen_output, target):
        # 生成したものが1(本物)だとだませたか
        # gan_loss = self.loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
        # mean absolute error 平均絶対誤差
        # l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
        gen_loss = self.loss_object(target, gen_output)
        # total_gen_loss = (self.lambda_val * gan_loss)
        return gen_loss

    # def discriminator_loss(self, disc_real_output, disc_generated_output):
    #     # 本物に対して1(本物)と判定できたか
    #     real_loss = self.loss_object(tf.ones_like(disc_real_output), disc_real_output)
    #     # 偽物に対して0(偽物)と判定できたか
    #     generated_loss = self.loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
    #     # lossの合計
    #     total_disc_loss = real_loss + generated_loss
    #     return total_disc_loss

    # # ===== model save =====
    # def save_generator(self, path):
    #     self.gen.save(path)

    # ===== 値変換系 =====
    def binary_from_img(self, data):
        return tf.greater_equal(data, 255)

    def binary_from_data(self, data):
        return tf.greater_equal(data, 0.5)

    def img_from_netdata(self, data):
        return data * 255

    def netdata_from_img(self, data):
        return data / 255
    # def binary_from_img(self, data):
    #     return tf.greater_equal(data, 127.5)
    #
    # def binary_from_data(self, data):
    #     return tf.greater_equal(data, 0)
    #
    # def img_from_netdata(self, data):
    #     return (data + 1) * 127.5
    #
    # def netdata_from_img(self, data):
    #     return (data / 127.5) - 1

    # ==============================================================================================================
    # ========== train関数 =========================================================================================
    # ==============================================================================================================
    @tf.function  # 関数をグラフモードで実行 https://www.tensorflow.org/guide/autograph?hl=ja
    def multi_train_step(self, ds_iter, device_list):
        generator_gradients_list = []
        discriminator_gradients_list = []
        for gpu_index in device_list:
            with tf.device('/gpu:%d' % gpu_index): # gpu単位の処理
                # tf.GradientTape()以下に勾配算出対象の計算を行う https://qiita.com/propella/items/5b2182b3d6a13d20fefd
                with tf.GradientTape() as gen_tape:
                    input_image, target = next(ds_iter)
                    target = nfunc.target_cut_padding(target=target, padding=self.get_padding())
                    # Generatorによる画像生成
                    generator = self.get_generator()
                    gen_output = generator(input_image, training=True)
                    # Discriminatorによる判定
                    # discriminator = self.get_discriminator()
                    # disc_real_output = discriminator([input_image, target], training=True)
                    # disc_generated_output = discriminator([input_image, gen_output], training=True)
                    # loss算出
                    gen_loss = self.generator_loss(gen_output, target)
                    # disc_loss = self.discriminator_loss(disc_real_output, disc_generated_output)
                # 勾配算出 trainable_variables:訓練可能な変数
                generator_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
                # discriminator_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
                # 後で平均を取る為に保存
                generator_gradients_list.append(generator_gradients)
                # discriminator_gradients_list.append(discriminator_gradients)
                # gpu単位の処理ここまで
        # with tf.device('/gpu:%d' % device_list[0]):  # gpu単位の処理
        generator = self.get_generator()
        discriminator = self.get_discriminator()
        # 勾配の平均、怪しい
        generator_gradients_average = nfunc.average_gradients(generator_gradients_list)
        # discriminator_gradients_average = nfunc.average_gradients(discriminator_gradients_list)
        # 勾配の適用
        self.get_generator_optimizer().apply_gradients(zip(generator_gradients_average, generator.trainable_variables))
        # self.get_discriminator_optimizer().apply_gradients(zip(discriminator_gradients_average, discriminator.trainable_variables))
        return

    @tf.function  # 関数をグラフモードで実行 https://www.tensorflow.org/guide/autograph?hl=ja
    def train_step(self, ds_iter, device_list, rate=1):
        with tf.device('/gpu:%d' % device_list[0]): # gpu単位の処理
            # tf.GradientTape()以下に勾配算出対象の計算を行う https://qiita.com/propella/items/5b2182b3d6a13d20fefd
            with tf.GradientTape() as gen_tape:
                input_image, target = next(ds_iter)
                target = nfunc.target_cut_padding(target=target, padding=self.get_padding())
                # Generatorによる画像生成
                generator = self.get_generator()
                gen_output = generator(input_image, training=True)
                # Discriminatorによる判定
                # discriminator = self.get_discriminator()
                # disc_real_output = discriminator([input_image, target], training=True)
                # disc_generated_output = discriminator([input_image, gen_output], training=True)
                # loss算出
                gen_loss = self.generator_loss(gen_output, target)
                # disc_loss = self.discriminator_loss(disc_real_output, disc_generated_output)
            # 勾配算出 trainable_variables:訓練可能な変数
            generator_gradients = gen_tape.gradient(gen_loss,generator.trainable_variables)
            # discriminator_gradients = disc_tape.gradient(disc_loss,discriminator.trainable_variables)
            # バッチサイズ毎のデータサイズの割合を勾配に適用させる 1/len(device_list)を掛けて、複数GPU時の時との差を調整
            generator_gradients = nfunc.rate_multiply(generator_gradients, rate * (1/len(device_list)))
            # discriminator_gradients = nfunc.rate_multiply(discriminator_gradients, rate * (1/len(device_list)))
            # gpu単位の処理ここまで
        # with tf.device('/gpu:%d' % gpu_index):  # gpu単位の処理
        generator = self.get_generator()
        discriminator = self.get_discriminator()
        # 勾配の適用
        self.get_generator_optimizer().apply_gradients(zip(generator_gradients, generator.trainable_variables))
        # self.get_discriminator_optimizer().apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))
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
# ========== NET ===========
    def _Generator(self):
        down_stack = [
            self._downsample(64, 4, apply_batchnorm=False),  # (bs, 128, 128, 64)
            self._downsample(128, 4),  # (bs, 64, 64, 128)
            self._downsample(256, 4),  # (bs, 32, 32, 256)
            self._downsample(512, 4),  # (bs, 16, 16, 512)
            self._downsample(512, 4),  # (bs, 8, 8, 512)
            self._downsample(512, 4),  # (bs, 4, 4, 512)
            self._downsample(512, 4),  # (bs, 2, 2, 512)
            self._downsample(512, 4),  # (bs, 1, 1, 512)
        ]
        up_stack = [
            self._upsample(512, 4, apply_dropout=True),  # (bs, 2, 2, 1024)
            self._upsample(512, 4, apply_dropout=True),  # (bs, 4, 4, 1024)
            self._upsample(512, 4, apply_dropout=True),  # (bs, 8, 8, 1024)
            self._upsample(512, 4),  # (bs, 16, 16, 1024)
            self._upsample(256, 4),  # (bs, 32, 32, 512)
            self._upsample(128, 4),  # (bs, 64, 64, 256)
            self._upsample(64, 4),  # (bs, 128, 128, 128)
        ]
        initializer = tf.random_normal_initializer(0., 0.02)
        last = tf.keras.layers.Conv2DTranspose(self.output_channel, 4,
                                               strides=2,
                                               padding='same',
                                               kernel_initializer=initializer,
                                               activation='sigmoid')  # (bs, 256, 256, 3) tanh->=-1~1
        concat = tf.keras.layers.Concatenate() # 連結
        inputs = tf.keras.layers.Input(shape=[None,None,3])
        x = inputs
        # Downsampling through the model
        skips = []
        for down in down_stack:
            x = down(x)
            skips.append(x)
        skips = reversed(skips[:-1])
        # Upsampling and establishing the skip connections
        for up, skip in zip(up_stack, skips):
            x = up(x)
            x = concat([x, skip])
        x = last(x)
        return tf.keras.Model(inputs=inputs, outputs=x)  # 返却値は-1~1

    def _Discriminator(self):
        initializer = tf.random_normal_initializer(0., 0.02)
        inp = tf.keras.layers.Input(shape=[None, None, 3], name='input_image')
        tar = tf.keras.layers.Input(shape=[None, None, 1], name='target_image')
        x = tf.keras.layers.concatenate([inp, tar])  # (bs, 256, 256, channels*2)
        down1 = self._downsample(64, 4, False)(x)  # (bs, 128, 128, 64)
        down2 = self._downsample(128, 4)(down1)  # (bs, 64, 64, 128)
        down3 = self._downsample(256, 4)(down2)  # (bs, 32, 32, 256)
        zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (bs, 34, 34, 256)
        conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                      kernel_initializer=initializer,
                                      use_bias=False)(zero_pad1)  # (bs, 31, 31, 512)
        batchnorm1 = tf.keras.layers.BatchNormalization()(conv)
        leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)
        zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (bs, 33, 33, 512)
        last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                      kernel_initializer=initializer)(zero_pad2)  # (bs, 30, 30, 1)
        return tf.keras.Model(inputs=[inp, tar], outputs=last)

# ============ NET FUNCTION ==========
    def _downsample(self, filters, size, apply_batchnorm=True):
        initializer = tf.random_normal_initializer(0., 0.02)
        result = tf.keras.Sequential()
        result.add(
            tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                                   kernel_initializer=initializer, use_bias=False))
        if apply_batchnorm:
            result.add(tf.keras.layers.BatchNormalization())
        result.add(tf.keras.layers.LeakyReLU())
        return result

    def _upsample(self, filters, size, apply_dropout=False):
        initializer = tf.random_normal_initializer(0., 0.02)
        result = tf.keras.Sequential()
        result.add(
            tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                            padding='same',
                                            kernel_initializer=initializer,
                                            use_bias=False))
        result.add(tf.keras.layers.BatchNormalization())
        if apply_dropout:
            result.add(tf.keras.layers.Dropout(0.5))
        result.add(tf.keras.layers.ReLU())
        return result
