# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import math
import gc
import csv
import time as timefunc
import sys
from tqdm import tqdm


def evenization(n):  # 偶数化
    return n + 1 if n % 2 != 0 else n


class ImageGenerator:
    def __init__(self, Generator_model, model_h, model_w, padding, fin_activate='tanh', use_gpu=True):
        self.model_h = model_h
        self.model_w = model_w
        self.padding = padding
        self.time_basis = 54000  # timeの基準
        self.day_time = 86400  # 1日
        self.fin_activate = fin_activate
        self.device = '/gpu:0' if use_gpu else '/cpu:0'
        self.Generator = tf.keras.models.load_model(Generator_model)
        print('Loading Generator')
        if use_gpu:
            with tf.device(self.device):
                zero_data = np.zeros(shape=[1, self.model_h, self.model_w, 3], dtype=np.float32)  # 空データ
                _ = self.generate_img(zero_data)  # GPUを使う場合,あらかじめGenerateを起動しておき,cudnnを先に呼び出しておく,と僅かに早くなる,気がする
        print('End Loading')

    def binary_from_data(self, data, fin_activate):
        if fin_activate is 'tanh':
            return tf.greater_equal(data, 0)
        if fin_activate is 'sigmoid':
            return tf.greater_equal(data, 0.5)

    def img_from_netdata(self, data, fin_activate):
        if fin_activate is 'tanh':
            return (data + 1) * 127.5
        if fin_activate is 'sigmoid':
            return data * 255

    def netdata_from_img(self, data, fin_activate):
        if fin_activate is 'tanh':
            return (data / 127.5) - 1
        if fin_activate is 'sigmoid':
            return data / 255

    def check_crop_index(self, norm_img):
        norm_h, norm_w, _ = norm_img.shape
        h_count = int((norm_h - self.padding * 2) / self.model_h)
        w_count = int((norm_w - self.padding * 2) / self.model_w)
        crop_h = [n // w_count for n in range(h_count * w_count)]
        crop_w = [n % w_count for n in range(h_count * w_count)]
        crop_top = [n * self.model_h for n in crop_h]
        crop_left = [n * self.model_w for n in crop_w]
        crop_index = list(zip(*[crop_top, crop_left]))
        return crop_index

    def _img_size_norm(self, img):
        origin_h, origin_w, _ = img.shape

        # 切り取る枚数を計算(切り取り枚数は偶数とする)
        sheets_h = evenization(math.ceil(origin_h / self.model_h))  # math.ceil 切り上げ
        sheets_w = evenization(math.ceil(origin_w / self.model_w))  # math.ceil 切り上げ

        # 必要となる画像サイズを求める(外側切り出しを考慮してpadding*2を足す)
        flame_h = sheets_h * self.model_h + (self.padding * 2)  # 偶数 * N * 偶数 = 偶数
        flame_w = sheets_w * self.model_w + (self.padding * 2)

        # 追加すべき画素数を求める
        extra_h = flame_h - origin_h  # if 偶数 - 奇数 = 奇数
        extra_w = flame_w - origin_w  # elif 偶数 - 偶数 = 偶数

        # 必要画素数のフレームを作って中心に画像を挿入
        flame = np.zeros([flame_h, flame_w, 3], dtype=np.uint8)
        top, bottom = math.floor(extra_h / 2), flame_h - math.ceil(extra_h / 2)  # 追加画素が奇数なら下右側に追加させるceil
        left, right = math.floor(extra_w / 2), flame_w - math.ceil(extra_w / 2)  # 追加画素が奇数なら下右側に追加させるceil
        flame[top:bottom, left:right, :] = img
        return flame, [top, bottom, left, right]

    @tf.function  # 関数をグラフモードで実行 https://www.tensorflow.org/guide/autograph?hl=ja
    def generate_img(self, data):
        out = self.Generator(data, training=False)
        return self.binary_from_data(out, self.fin_activate)

    def run(self, img_path, out_path, batch=50, time_out_path=None):

        # プログラム稼働開始時刻取得
        time_stock = []
        time_stock.append(timefunc.time())

        # ========== 画像の読み込み read_time ==========
        print('--- Loading Image ---')
        img_byte = tf.io.read_file(img_path)
        in_img = tf.image.decode_png(img_byte, channels=3)
        time_stock.append(timefunc.time())  # 画像の読み込み終了時刻取得

        # ========== データセット作成 ds_time ==========
        print('--- Dataset Making ---')
        origin_h, origin_w, _ = in_img.shape  # 読み込み画像のサイズ取得
        in_img = np.array(in_img, dtype=np.uint8)  #
        norm_img, size = self._img_size_norm(in_img)  # 画像サイズを調整する

        # 元画像を開放する
        del in_img
        gc.collect()

        # 切り出す画像の座標を求める
        crop_index = self.check_crop_index(norm_img)

        # 画像切り出し用関数
        def _cut_img(img_index):
            cut = tf.slice(norm_img, [img_index[0], img_index[1], 0],
                           [self.model_h + (self.padding * 2), self.model_w + (self.padding * 2), 3])
            cut = tf.cast(cut, tf.float32)
            cut = self.netdata_from_img(cut, fin_activate=self.fin_activate)
            return cut

        # データセット
        dataset = tf.data.Dataset.from_tensor_slices(crop_index)
        dataset = dataset.map(_cut_img)
        dataset = dataset.batch(batch)

        time_stock.append(timefunc.time())  # データセット定義終了時刻取得

        # ========== 画像生成 gen_time ==========
        print('--- Generate Image ---')
        norm_h, norm_w, _ = norm_img.shape
        out_flame = np.zeros(shape=[norm_h - self.padding * 2, norm_w - self.padding * 2, 1], dtype=np.float32)
        data_iter = iter(dataset)
        index = 0
        net_time = 0
        with tqdm(total=math.ceil(len(crop_index)), desc='image generate') as pbar:
            for data in data_iter:
                net_start_time = timefunc.time()
                with tf.device(self.device):
                    outset = self.generate_img(data)
                net_time += timefunc.time() - net_start_time
                for out in outset:
                    crop_top, crop_left = crop_index[index]
                    out_flame[crop_top: crop_top + self.model_h,
                    crop_left: crop_left + self.model_h,
                    0] = out[:, :, 0]
                    index += 1
                pbar.update(batch)  # プロセスバーを進行
        out_img = self.img_from_netdata(out_flame, fin_activate=self.fin_activate)
        out_img = out_img[size[0]:size[1], size[2]:size[3]]
        time_stock.append(timefunc.time())  # 画像生成終了時刻取得

        # ========== 画像出力 out_time ==========
        print('--- Output Image ---')
        out_img = tf.cast(out_img, tf.uint8)
        out_byte = tf.image.encode_png(out_img)
        tf.io.write_file(filename=out_path, contents=out_byte)
        time_stock.append(timefunc.time())  # 画像出力終了時刻取得

        # ---------- 時間計測結果 ---------
        if time_out_path is not None:
            time_num_set = []
            time_str_set = []
            csv_header = ['read_time', 'ds_time', 'gen_time', 'out_time', 'total_time', 'only_net_time']
            for i in range(len(time_stock) - 1):
                time_num_set.append(time_stock[i + 1] - time_stock[i] + self.time_basis)
            time_num_set.append(time_stock[-1] - time_stock[0] + self.time_basis)
            time_num_set.append(net_time + self.time_basis)
            for i in range(len(time_num_set)):
                ms_time, s_time = math.modf(time_num_set[i])  # ミニセカンド セカンド
                day, times = divmod(s_time, self.day_time)  # 日数と時間に
                day = int(day)
                step_times = timefunc.strptime(timefunc.ctime(times))
                str_time = str(day) + 'd' + timefunc.strftime("%H:%M:%S", step_times) + str(ms_time)[1:]
                time_str_set.append(str_time)
                print(csv_header[i] + ': ' + str_time)
            with open(time_out_path, 'w') as f:
                writer = csv.writer(f, lineterminator='\n')
                writer.writerow(csv_header)
                writer.writerow(time_str_set)
