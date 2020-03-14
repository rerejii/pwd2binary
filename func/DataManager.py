import tensorflow as tf

class DataManager():
    def __init__(self, tfrecord_path, img_root, batch_size,
                 net_cls, data_n, cycle_length=4, suffle_buffer=-1,
                 data_cls=3, use_aveimg=False):
        self.data_cls = data_cls
        self.img_root = img_root
        self.net_cls = net_cls
        # データセットの読み込み及び整形
        _dataset = tf.data.Dataset.list_files(tfrecord_path)
        _dataset = _dataset.interleave(tf.data.TFRecordDataset,
                                       cycle_length=cycle_length,
                                       num_parallel_calls=tf.data.experimental.AUTOTUNE)

        if use_aveimg==False:
            _dataset = _dataset.map(self._parse_function)
            if (suffle_buffer!=-1):
                _dataset = _dataset.shuffle(int(suffle_buffer))
            _dataset = _dataset.map(self._image_from_path)
            _dataset = _dataset.map(self._netdata_normalize)
        else:
            _dataset = _dataset.map(self._gray_parse_function)
            if (suffle_buffer != -1):
                _dataset = _dataset.shuffle(int(suffle_buffer))
            _dataset = _dataset.map(self._gray_image_from_path)
            _dataset = _dataset.map(self._gray_netdata_normalize)

        _dataset = _dataset.batch(batch_size)
        self.dataset = _dataset
        self.batch_size = batch_size
        self.max_data_n = data_n
        self.remain_data_n = 0
        self.ds_iter = None


    def get_inited_iter(self):
        self.remain_data_n = self.max_data_n
        return iter(self.dataset)

    def data_used_apply(self, use_count=1):
        self.remain_data_n = self.remain_data_n - (use_count * self.batch_size)
        if self.remain_data_n < 0:
            self.remain_data_n = 0

    def get_remain_data(self):
        return self.remain_data_n

    def get_batch_size(self):
        return self.batch_size

    def get_total_data(self):
        return self.max_data_n

    def get_next_data_rate(self):
        if self.remain_data_n > self.batch_size:
            return 1
        else:
            return self.remain_data_n / self.batch_size

    def get_next_data_n(self):
        if self.remain_data_n > self.batch_size:
            return self.batch_size
        else:
            return self.remain_data_n

    # tfrecordsの解析読み込み関数
    def _parse_function(self, _data):
        _features = {'sample_path': tf.io.FixedLenFeature((), tf.string, default_value=''),
                    'answer_path': tf.io.FixedLenFeature((), tf.string, default_value=''),}  # dictの型を作る
        _parsed_features = tf.io.parse_single_example(serialized=_data, features=_features)  # データを解析しdictにマッピングする
        return _parsed_features  # 解析したデータを返す

    # パスから画像を読みだす
    def _image_from_path(self, _parsed_features):
        _sample_png_bytes = tf.io.read_file(self.img_root + '/' + _parsed_features['sample_path'])
        _sample_image = tf.image.decode_png(_sample_png_bytes, channels=self.data_cls, dtype=tf.dtypes.uint8,)
        _answer_png_bytes = tf.io.read_file(self.img_root + '/' + _parsed_features['answer_path'])
        _answer_image = tf.image.decode_png(_answer_png_bytes, channels=1, dtype=tf.dtypes.uint8,)
        return _sample_image, _answer_image

    # normalizing the images
    def _netdata_normalize(self, _sample_image, _answer_image):
        _sample_image = tf.cast(_sample_image, tf.float32)
        _answer_image = tf.cast(_answer_image, tf.float32)
        _sample_image = self.net_cls.netdata_from_img(_sample_image)
        _answer_image = self.net_cls.netdata_from_img(_answer_image)
        return _sample_image, _answer_image

    # ===== gray =====
    # tfrecordsの解析読み込み関数
    def _gray_parse_function(self, _data):
        _features = {'sample_path': tf.io.FixedLenFeature((), tf.string, default_value=''),
                    'answer_path': tf.io.FixedLenFeature((), tf.string, default_value=''),
                    'average_path': tf.io.FixedLenFeature((), tf.string, default_value=''), }  # dictの型を作る
        _parsed_features = tf.io.parse_single_example(serialized=_data, features=_features)  # データを解析しdictにマッピングする
        return _parsed_features  # 解析したデータを返す

    # パスから画像を読みだす
    def _gray_image_from_path(self, _parsed_features):
        _sample_png_bytes = tf.io.read_file(self.img_root + '/' + _parsed_features['sample_path'])
        _sample_image = tf.image.decode_png(_sample_png_bytes, channels=self.data_cls, dtype=tf.dtypes.uint8,)
        _answer_png_bytes = tf.io.read_file(self.img_root + '/' + _parsed_features['answer_path'])
        _answer_image = tf.image.decode_png(_answer_png_bytes, channels=1, dtype=tf.dtypes.uint8,)
        _average_png_bytes = tf.io.read_file(self.img_root + '/' + _parsed_features['average_path'])
        _average_image = tf.image.decode_png(_average_png_bytes, channels=1, dtype=tf.dtypes.uint8, )
        return _sample_image, _answer_image, _average_image

    # normalizing the images
    def _gray_netdata_normalize(self, _sample_image, _answer_image, _average_image):
        _sample_image = tf.cast(_sample_image, tf.float32)
        _answer_image = tf.cast(_answer_image, tf.float32)
        _average_image = tf.cast(_average_image, tf.float32)
        _sample_image = self.net_cls.netdata_from_img(_sample_image)
        _answer_image = self.net_cls.netdata_from_img(_answer_image)
        _average_image = self.net_cls.netdata_from_img(_average_image)
        return _sample_image, _answer_image, _average_image
