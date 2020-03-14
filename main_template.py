# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function, unicode_literals

import os
import numpy as np
import pandas as pd
import sys
import random
from tqdm import tqdm
from importlib import import_module

# ========== import設定 ==========
work_dirname = (os.path.dirname(os.path.abspath(__file__)))
sys.path.append(work_dirname + '/' + 'net_cls/')
sys.path.append(work_dirname + '/' + 'func/')

# ========== GPU設定 ==========
args = sys.argv
DEVICE_NUMBER_STR = args[1]  # 使用するGPU設定
# device番号のlistを生成(複数GPU学習時の割り当て調整に使用)
DEVICE_STR_LIST = DEVICE_NUMBER_STR.split(',')
DEVICE_LIST = [int(n) for n in DEVICE_STR_LIST]
# 環境変数の調整
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = DEVICE_NUMBER_STR

# ========== TensorFlow呼び出し ==========
# 環境変数の変更を適用したのちに呼び出す
import tensorflow as tf

# ========== 乱数シード指定 ==========
seed = 1
os.environ['PYTHONHASHSEED'] = str(seed)  # https://docs.python.org/ja/3.6/using/cmdline.html#envvar-PYTHONHASHSEED
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# ========== 定数1(ほぼ毎回変更するシリーズ) ==========
# ネットワークのpaddingサイズも調整したか？
# A B C D ...
DATASET = ''
# 出力ネーム
MAINNAME = ''
# importする net cls
IMPORT_NET_CLS = ''
# sigmoid tanh ...
FIN_ACTIVATE = ''
#  /home/username/work/myTensor/dataset2/DR*/
TFRECORD_FOLDER = '' + DATASET
#  /home/username/work/myTensor/dataset2/Img*/
IMAGE_ROOT_PATH = ''
#  '20200223'
OUT_ROOT_FOLDER_NAME = ''
#  '20200223'
EPOCH_OUT_ROOT_FOLDER_NAME = ''

# ========== 定数2(たまに修正するシリーズ) ==========
GENERATOR_TEST_PATH = '/home/username/work/myTensor/dataset2/17H-0863-1_L0002-new/L0002.png'
GENERATOR_OUTPATH = OUT_ROOT_FOLDER + '/out.png'
# MAINNAME+'-'+DATASET+'.h5'
CYCLE_LENGHT = 4  # データ読み込み処理オーバーラップ数
SHUF_LEARN_BATCH_SIZE = 3
LEARN_BATCH_SIZE = 50
VALIDATION_BATCH_SIZE = 50
TEST_BATCH_SIZE = 50
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
SUFFLE_BUFFER_SIZE = 100000
OUTPUT_CHANNELS = 1
CHECK_STEP = 24000
END_EPOCH = int(args[2])

# ========== 調整 ==========
DISKNAME = args[3]
OUT_ROOT_FOLDER = '/home/hdd/'+DISKNAME+'/username/' + OUT_ROOT_FOLDER_NAME + '/' + MAINNAME+'/'+MAINNAME+ '-' + DATASET
GENERATOR_OUTPATH = OUT_ROOT_FOLDER + '/out.png'
EPOCH_OUT_ROOT_FOLDER = '/home/hdd/'+DISKNAME+'/username/' + EPOCH_OUT_ROOT_FOLDER_NAME + '_epoch' + '/' + MAINNAME +\
                        '/'+MAINNAME+ '-' + DATASET if EPOCH_OUT_ROOT_FOLDER_NAME is not '' else None

# ========== 確認用画像集 ==========
CHECK_SAMPLE_PATH = [

]
CHECK_ANSWER_PATH = [

]
CHECK_IMG_TAG = [

] # 'test', 'valid', 'learn'
check_img_path = []
for i in range(len(CHECK_SAMPLE_PATH)):
    check_img_path.append([IMAGE_ROOT_PATH+'/'+CHECK_SAMPLE_PATH[i],
                           IMAGE_ROOT_PATH+'/'+CHECK_ANSWER_PATH[i],
                           CHECK_IMG_TAG[i],
                           ],)

# ========== My class inport ==========
# from EncodeManager import NetManager  # 呼び出すネットによって変える
NetManager = import_module(IMPORT_NET_CLS).NetManager
from DataManager import DataManager
from PathManager import PathManager
from ImageGenerator import ImageGenerator
from FitManager import FitManager
import ShareNetFunc as nfunc
import PlotFunc

# ========== NetManager呼び出し ==========
net_cls = NetManager()

# ========== PathManager呼び出し ==========
path_cls = PathManager(tfrecord_folder=TFRECORD_FOLDER,
                       output_rootfolder=OUT_ROOT_FOLDER,
                       epoch_output_rootfolder=EPOCH_OUT_ROOT_FOLDER)
path_cls.all_makedirs()  # 結果保存フォルダ生成

# ========== DataSet呼び出し ==========
# プロパティデータ読み込み
df = pd.read_csv(path_cls.get_property_path())
shuf_train_ds_cls = DataManager(tfrecord_path=path_cls.get_train_ds_path(),
                                img_root=IMAGE_ROOT_PATH,
                                batch_size=SHUF_LEARN_BATCH_SIZE,
                                net_cls=net_cls,
                                data_n=df.at[0, 'total_learn_data'],
                                suffle_buffer=SUFFLE_BUFFER_SIZE,
                                )
train_ds_cls = DataManager(tfrecord_path=path_cls.get_train_ds_path(),
                           img_root=IMAGE_ROOT_PATH,
                           batch_size=LEARN_BATCH_SIZE,
                           net_cls=net_cls,
                           data_n=df.at[0, 'total_learn_data'],
                           )
test_ds_cls = DataManager(tfrecord_path=path_cls.get_test_ds_path(),
                          img_root=IMAGE_ROOT_PATH,
                          batch_size=TEST_BATCH_SIZE,
                          net_cls=net_cls,
                          data_n=df.at[0, 'total_test_data'],
                          )
valid_ds_cls = DataManager(tfrecord_path=path_cls.get_validation_ds_path(),
                          img_root=IMAGE_ROOT_PATH,
                          batch_size=VALIDATION_BATCH_SIZE,
                          net_cls=net_cls,
                          data_n=df.at[0, 'total_valid_data'],
                          )

# ========== ネットワーク呼び出し ==========
fit_cls = FitManager(net_cls=net_cls,
                     path_cls=path_cls,
                     shuf_train_ds_cls=shuf_train_ds_cls,
                     train_ds_cls=train_ds_cls,
                     test_ds_cls=test_ds_cls,
                     validation_ds_cls=valid_ds_cls,
                     check_img_path=check_img_path,
                     )

# ========== 学習＆評価 ==========
fit_cls.fit(
    end_epoch=END_EPOCH,
    device_list=DEVICE_LIST,
    ckpt_step=CHECK_STEP,
)

# ========== plot ==========
PlotFunc.accuracy_plot(
    path_cls=path_cls,
    title='[TEST]_' + os.path.basename(OUT_ROOT_FOLDER),
    label=DATASET,
    item='test_accuracy',
)
PlotFunc.accuracy_plot(
    path_cls=path_cls,
    title='[TRAIN]_' + os.path.basename(OUT_ROOT_FOLDER),
    label=DATASET,
    item='train_accuracy',
)

# ========== save model ==========
modelpath = path_cls.make_model_path(MAINNAME + '-' + DATASET + '.h5')
nfunc.save_best_generator_model(net_cls=net_cls, path_cls=path_cls, path=modelpath)

# ========== 生成速度計測 ==========
gen_cls = ImageGenerator(Generator_model=modelpath,
                        model_h=IMAGE_HEIGHT,
                        model_w=IMAGE_WIDTH,
                        fin_activate=FIN_ACTIVATE,
                        padding=net_cls.get_padding())
gen_cls.run(img_path=GENERATOR_TEST_PATH,
            out_path=GENERATOR_OUTPATH,
            time_out_path=path_cls.make_csv_path('Generator_time.csv'))
gen_cls = ImageGenerator(Generator_model=modelpath,
                        model_h=IMAGE_HEIGHT,
                        model_w=IMAGE_WIDTH,
                        fin_activate=FIN_ACTIVATE,
                        padding=net_cls.get_padding(),
                        use_gpu=False)
gen_cls.run(img_path=GENERATOR_TEST_PATH,
            out_path=GENERATOR_OUTPATH,
            time_out_path=path_cls.make_csv_path('Generator_time_cpu.csv'),)



