import tensorflow as tf
import sys
from tqdm import tqdm
# from tqdm import tqdm_notebook as tqdm
import shutil
import time
import csv
import os
from datetime import datetime
import numpy as np

black = [0, 0, 0]  # 予測:0黒 解答:0黒
white = [255, 255, 255]  # 予測:1白 解答:1白
red = [255, 0, 0]  # 予測:1白 解答:0黒
blue = [0, 204, 255]  # 予測:0黒 解答:1白


# ========== only train ==========
def average_gradients(tower_grads, rate=1):
    average_grads = []
    for grad in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   (grad0_gpu0, ... , grad0_gpuN)
        grad = tf.reduce_mean(grad, 0)
        # rate = tf.cast(data_rate, dtype=tf.float32)
        grad = tf.multiply(grad, rate)
        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        average_grads.append(grad)
    return average_grads


def rate_multiply(grads, rate):
    out_grads = []
    for grad in grads:
        grad = tf.multiply(grad, rate)
        out_grads.append(grad)
    return out_grads


# ========== only save ==========
# stock = [ [total_step, train_accuracy, test_accuracy, validation_accuracy], [...], ...]
def log_write(writer, stock, scalar_name, image_tag, check_img_path, path_cls):
    for data in stock:
        step = data[0]
        image_list = [[] for i in range(len(image_tag))]
        for pi in range(len(check_img_path)):
            target_path, answer_path, tag = check_img_path[pi]
            read_path = make_step_img_path(step=step, img_path=target_path, path_cls=path_cls)
            img = image_from_path(read_path)
            for ti in range(len(image_tag)):
                image_list[ti].append(img) if tag == image_tag[ti] else None
        with writer.as_default():  # ログに書き込むデータはここに記載するらしい？ https://www.tensorflow.org/api_docs/python/tf/summary
            for di in range(len(data)-1):
                tf.summary.scalar(name=scalar_name[di], data=data[di+1], step=step)
            for ti in range(len(image_tag)):
                tf.summary.image(name=image_tag[ti], data=image_list[ti], step=step, max_outputs=10) if image_list[ti] else None


def accuracy_log_write(writer, stock, scalar_name):
    for data in stock:
        step = data[0]
        with writer.as_default():  # ログに書き込むデータはここに記載するらしい？ https://www.tensorflow.org/api_docs/python/tf/summary
            for di in range(len(data)-1):
                tf.summary.scalar(name=scalar_name[di], data=data[di+1], step=step)


# csvに書き込み
def write_csv(path_cls, filename, datalist):
    with open(path_cls.make_csv_path(filename=filename), 'a') as f:
        for data in datalist:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow(data)


# ========== 評価系 ==========
# 評価
def evaluate(net_cls, out, ans, region=[], batch_data_n=1):
    out_binary = net_cls.binary_from_data(out)
    ans_binary = net_cls.binary_from_data(ans)
    correct_prediction = tf.equal(out_binary, ans_binary)
    if not region:
        return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    region_result = []
    max_region = region[-1]
    cut_o = int((max_region - region[0]) / 2)
    cut_out = out_binary[:, cut_o:max_region-cut_o, cut_o:max_region-cut_o, :]
    cut_ans = ans_binary[:, cut_o:max_region-cut_o, cut_o:max_region-cut_o, :]
    region_result.append(tf.reduce_mean(tf.cast(tf.equal(cut_out, cut_ans), tf.float32)))
    for i in range(len(region)-1):
        cut_o = int((max_region - region[i+1]) / 2)
        cut_i = int((max_region - region[i]) / 2)
        cut_out = out_binary[:, cut_o:max_region-cut_o, cut_o:max_region-cut_o]
        cut_ans = ans_binary[:, cut_o:max_region-cut_o, cut_o:max_region-cut_o]
        fil = np.ones((max_region-cut_o*2, max_region-cut_o*2))
        fil[cut_i:max_region-cut_i, cut_i:max_region-cut_i] = 0
        fil = tf.reshape(fil, [1, max_region-cut_o*2, max_region-cut_o*2, 1])
        fil = tf.tile(fil, [batch_data_n, 1, 1, 1])
        tf_fil = tf.where(fil)
        cut_out = tf.gather_nd(cut_out, tf_fil)
        cut_ans = tf.gather_nd(cut_ans, tf_fil)
        region_result.append(tf.reduce_mean(tf.cast(tf.equal(cut_out, cut_ans), tf.float32)))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32)), region_result





# 評価画像の出力
def evalute_img(net_cls, out, ans):
    out_binary = net_cls.binary_from_data(out)
    out_shape = out_binary.numpy().shape
    result_img = np.zeros([out_shape[0], out_shape[1], 3])
    out_3d = out_binary.numpy().repeat(3).reshape(out_shape[0], out_shape[1], 3)
    ans_binary = net_cls.binary_from_img(ans)
    if (tf.rank(ans_binary)) == 2:
        ans_3d = ans_binary.numpy().repeat(3).reshape(out_shape[0], out_shape[1], 3)
    else:
        ans_3d = ans_binary.numpy()
    result_img += (out_3d == 0) * (ans_3d == 0) * black  # 黒 正解
    result_img += (out_3d == 1) * (ans_3d == 1) * white  # 白 正解
    result_img += (out_3d == 1) * (ans_3d == 0) * red  # 赤 黒欠け
    result_img += (out_3d == 0) * (ans_3d == 1) * blue  # 青 黒余分
    return result_img.astype(np.uint8)


def img_check(step, net_cls, path_cls, check_img_path):
    generator = net_cls.get_generator()
    for i in range(len(check_img_path)):
        target_path, answer_path, tag = check_img_path[i]
        # Generatorによる画像生成
        sample_data = normalize_netdata_from_path(net_cls=net_cls, img_path=target_path)
        answer_data = image_from_path(answer_path, ch=1)
        answer_data = target_cut_padding(target=answer_data, padding=net_cls.get_padding(),
                                         batch_shape=False)
        gen_output = generator(sample_data[np.newaxis, :, :, :], training=True)
        eva_img = evalute_img(net_cls=net_cls, out=gen_output[0, :, :, :], ans=answer_data)
        encoder_img = tf.image.encode_png(eva_img)
        save_path = make_step_img_path(step=step, img_path=target_path, path_cls=path_cls)
        tf.io.write_file(filename=save_path, contents=encoder_img)


# ========== 変換系 ==========
# ファイルパスからネットデータに正規化
def normalize_netdata_from_path(net_cls, img_path):
    png_bytes = tf.io.read_file(img_path)
    img = tf.image.decode_png(png_bytes, channels=3, dtype=tf.dtypes.uint8, )
    img = tf.cast(img, tf.float32)
    normalize_img = net_cls.netdata_from_img(img)
    return normalize_img


# ファイルパスから画像に
def image_from_path(img_path, ch=3):
    png_bytes = tf.io.read_file(img_path)
    img = tf.image.decode_png(png_bytes, channels=ch, dtype=tf.dtypes.uint8,)
    img = tf.cast(img, tf.float32)
    return img


# 検証用画像パスを生成する
def make_step_img_path(step, img_path, path_cls):
    img_name = os.path.splitext(os.path.basename(img_path))[0]  # ('Sample_X017_Y024', '.png')
    image_root_path = path_cls.get_image_folder_path()
    path = image_root_path + '/' + img_name + '/' + ('step-%s.png' % (step))
    os.makedirs(image_root_path + '/' + img_name, exist_ok=True)
    return path


# paddingサイズ分を切り出す
def color_cut_padding(img, padding):
    if padding == 0:
        return img
    if (tf.rank(img)) == 4:
        img = img[:, padding:-padding, padding:-padding, :]
    elif (tf.rank(img)) == 3:
        img = img[padding:-padding, padding:-padding, :]
    # elif (tf.rank(img)) is not 0:
    #     print('color_cut_padding ERROR!')
    #     print('rank:' + str(tf.rank(img)))
    #     sys.exit()
    return img


# paddingサイズ分を切り出す
def target_cut_padding(target, padding, batch_shape=True):
    if padding == 0:
        return target
    if (tf.rank(target)) == 4:
        target = target[:, padding:-padding, padding:-padding, :]
    elif (tf.rank(target)) == 3:
        if batch_shape:
            target = target[:, padding:-padding, padding:-padding]
        else:
            target = target[padding:-padding, padding:-padding, :]
    elif (tf.rank(target)) == 2:
        target = target[padding:-padding, padding:-padding]
    # elif (tf.rank(target)) is not 0:
    #     print('target_cut_padding ERROR!')
    #     print('rank:' + str(tf.rank(target)))
    #     sys.exit()
    return target


# ========== DeepOtsu ==========
def ave_color_img(mask, img, mask_repeat=1):
    # 1
    # mask3 = mask.repeat(mask_repeat).reshape(img.shape)
    # mask3 = tf.reshape(tf.repeat(mask, mask_repeat), img.shape)
    mask = tf.cast(mask, tf.float32)
    mask3 = tf.cast(mask, tf.float32)
    pass_img = img * mask3
    count_n = tf.reduce_sum(mask)
    color = []
    if mask_repeat is 1:
        color.append(tf.cast((tf.reduce_sum(pass_img[:, :, :]) / count_n), tf.uint8))
    else:
        for i in range(mask_repeat):
            color.append(tf.cast((tf.reduce_sum(pass_img[:, :, :, 0]) / count_n), tf.uint8))

    # 0
    # z_mask3 = tf.reshape(tf.repeat(mask == 0, mask_repeat), img.shape)
    z_mask3 = tf.cast((mask == 0), tf.float32)
    pass_img = img * mask3
    count_n = tf.reduce_sum(mask)
    z_color = []
    if mask_repeat is 1:
        z_color.append(tf.cast((tf.reduce_sum(pass_img[:, :, :]) / count_n), tf.uint8))
    else:
        for i in range(mask_repeat):
            z_color.append(tf.cast((tf.reduce_sum(pass_img[:, :, :, 0]) / count_n), tf.uint8))

    return tf.add((mask3 * color), (z_mask3 * z_color))


# ========== save model ==========
# モデル全体を１つのHDF5ファイルに保存します。***.h5
# https://www.tensorflow.org/tutorials/keras/save_and_load?hl=ja
def save_best_generator_model(net_cls, path_cls, path):
    path_list = path_cls.search_best_path(filename=path_cls.ckpt_step_name)
    if path_list:
        root, ext = os.path.splitext(path_list[0])
        net_cls.ckpt_restore(path=root)
    net_cls.get_generator().save(path)


# ========== val ==========
def get_black():
    return black


# ========== write log ==========
# def accuracy_csv2log(csv_path, out_path, scalar_name):
#     data = csv.reader()
#     writer = tf.summary.create_file_writer(logdir=out_path)
#     log_write(writer=writer, stock=data,
#               scalar_name=scalar_name, image_tag=summary_image_tag,
#               check_img_path=self.check_img_path, path_cls=self.path_cls)




def get_white():
    return white


def get_red():
    return red


def get_blue():
    return blue
