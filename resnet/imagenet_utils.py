# -*- coding: utf-8 -*-
# File: imagenet_utils.py


import cv2
import numpy as np
import tqdm
import multiprocessing
import tensorflow as tf
from abc import abstractmethod

from tensorpack import *
from tensorpack import ModelDesc
from tensorpack.input_source import QueueInput, StagingInput
from tensorpack.dataflow import (
    JoinData, imgaug, dataset, AugmentImageComponent, PrefetchDataZMQ,
    BatchData, MultiThreadMapData)
from tensorpack.predict import PredictConfig, FeedfreePredictor
from tensorpack.utils.stats import RatioCounter
from tensorpack.models import regularize_cost
from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.tfutils.common import get_global_step_var
from tensorpack.utils import logger
import ilsvrcsemi
from flip_gradient import flip_gradient

class GoogleNetResize(imgaug.ImageAugmentor):
    """
    crop 8%~100% of the original image
    See `Going Deeper with Convolutions` by Google.
    """
    def __init__(self, crop_area_fraction=0.08,
                 aspect_ratio_low=0.75, aspect_ratio_high=1.333,
                 target_shape=224):
        self._init(locals())

    def _augment(self, img, _):
        h, w = img.shape[:2]
        area = h * w
        for _ in range(10):
            targetArea = self.rng.uniform(self.crop_area_fraction, 1.0) * area
            aspectR = self.rng.uniform(self.aspect_ratio_low, self.aspect_ratio_high)
            ww = int(np.sqrt(targetArea * aspectR) + 0.5)
            hh = int(np.sqrt(targetArea / aspectR) + 0.5)
            if self.rng.uniform() < 0.5:
                ww, hh = hh, ww
            if hh <= h and ww <= w:
                x1 = 0 if w == ww else self.rng.randint(0, w - ww)
                y1 = 0 if h == hh else self.rng.randint(0, h - hh)
                out = img[y1:y1 + hh, x1:x1 + ww]
                out = cv2.resize(out, (self.target_shape, self.target_shape), interpolation=cv2.INTER_CUBIC)
                return out
        out = imgaug.ResizeShortestEdge(self.target_shape, interp=cv2.INTER_CUBIC).augment(img)
        out = imgaug.CenterCrop(self.target_shape).augment(out)
        return out


def fbresnet_augmentor(isTrain):
    """
    Augmentor used in fb.resnet.torch, for BGR images in range [0,255].
    """
    if isTrain:
        augmentors = [
            GoogleNetResize(),
            # It's OK to remove the following augs if your CPU is not fast enough.
            # Removing brightness/contrast/saturation does not have a significant effect on accuracy.
            # Removing lighting leads to a tiny drop in accuracy.
            
            imgaug.RandomOrderAug(
                [# We removed the following augmentation
                 #imgaug.BrightnessScale((0.6, 1.4), clip=False),
                 #imgaug.Contrast((0.6, 1.4), clip=False),
                 #imgaug.Saturation(0.4, rgb=False),
                 #rgb-bgr conversion for the constants copied from fb.resnet.torch
                 imgaug.Lighting(0.1,
                                 eigval=np.asarray(
                                     [0.2175, 0.0188, 0.0045][::-1]) * 255.0,
                                 eigvec=np.array(
                                     [[-0.5675, 0.7192, 0.4009],
                                      [-0.5808, -0.0045, -0.8140],
                                      [-0.5836, -0.6948, 0.4203]],
                                     dtype='float32')[::-1, ::-1]
                                 )]),
            imgaug.Flip(horiz=True),
        ]
    else:
        augmentors = [
            imgaug.ResizeShortestEdge(256, cv2.INTER_CUBIC),
            imgaug.CenterCrop((224, 224)),
        ]
    return augmentors


def get_imagenet_dataflow(
        datadir, name, batch_size,
        augmentors, parallel=None):
    """
    See explanations in the tutorial:
    http://tensorpack.readthedocs.io/en/latest/tutorial/efficient-dataflow.html
    """
    assert name in ['train', 'val', 'test']
    assert datadir is not None
    assert isinstance(augmentors, list)
    isTrain = name == 'train'
    if parallel is None:
        parallel = min(40, 16)  # assuming hyperthreading
    if isTrain:
        ds1 = ilsvrcsemi.ILSVRC12(datadir, name, shuffle=True, labeled=True)
        ds2 = ilsvrcsemi.ILSVRC12(datadir, name, shuffle=True, labeled=False)
        ds1 = AugmentImageComponent(ds1, augmentors, copy=False)
        ds2 = AugmentImageComponent(ds2, augmentors, copy=False)
        ds = JoinData([ds1, ds2])

        if parallel < 16:
            logger.warn("DataFlow may become the bottleneck when too few processes are used.")
        ds = PrefetchDataZMQ(ds, parallel)
        ds = BatchData(ds, batch_size, remainder=False)
    else:
        ds = dataset.ILSVRC12Files(datadir, name, shuffle=False)
        aug = imgaug.AugmentorList(augmentors)

        def mapf(dp):
            fname, cls = dp
            im = cv2.imread(fname, cv2.IMREAD_COLOR)
            im = aug.augment(im)
            return im, cls, im, cls
        ds = MultiThreadMapData(ds, parallel, mapf, buffer_size=2000, strict=True)
        ds = BatchData(ds, batch_size, remainder=True)
        ds = PrefetchDataZMQ(ds, 1)
    return ds


def eval_on_ILSVRC12(model, sessinit, dataflow):
    pred_config = PredictConfig(
        model=model,
        session_init=sessinit,
        input_names=['input', 'label', 'input2', 'label2'],
        output_names=['wrong-top1', 'wrong-top5']
    )
    acc1, acc5 = RatioCounter(), RatioCounter()

    # This does not have a visible improvement over naive predictor,
    # but will have an improvement if image_dtype is set to float32.
    pred = FeedfreePredictor(pred_config, StagingInput(QueueInput(dataflow), device='/gpu:0'))
    for _ in tqdm.trange(dataflow.size()):
        top1, top5 = pred()
        batch_size = top1.shape[0]
        acc1.feed(top1.sum(), batch_size)
        acc5.feed(top5.sum(), batch_size)

    print("Top1 Error: {}".format(acc1.ratio))
    print("Top5 Error: {}".format(acc5.ratio))


class ImageNetModel(ModelDesc):
    image_shape = 224

    """
    uint8 instead of float32 is used as input type to reduce copy overhead.
    It might hurt the performance a liiiitle bit.
    The pretrained models were trained with float32.
    """
    image_dtype = tf.uint8

    """
    Either 'NCHW' or 'NHWC'
    """
    data_format = 'NCHW'

    """
    Whether the image is BGR or RGB. If using DataFlow, then it should be BGR.
    """
    image_bgr = True

    weight_decay = 1e-4

    """
    To apply on normalization parameters, use '.*/W|.*/gamma|.*/beta'
    """
    weight_decay_pattern = '.*/W'

    """
    Scale the loss, for whatever reasons (e.g., gradient averaging, fp16 training, etc)
    """
    loss_scale = 1.

    """
    Label smoothing (See tf.losses.softmax_cross_entropy)
    """
    label_smoothing = 0.

    def inputs(self):
        return [tf.placeholder(self.image_dtype, [None, self.image_shape, self.image_shape, 3], 'input'),
                tf.placeholder(tf.int32, [None], 'label'),
                tf.placeholder(self.image_dtype, [None, self.image_shape, self.image_shape, 3], 'input2'),
                tf.placeholder(tf.int32, [None], 'label2')]
    def build_graph(self, image1, label1, image2, _):
        image1 = self.image_preprocess(image1)
        image2 = self.image_preprocess(image2)
        is_training = get_current_tower_context().is_training

        # Shuffle unlabeled data within batch
        if is_training:
            image2 = tf.random_shuffle(image2)
        
        assert self.data_format in ['NCHW', 'NHWC']
        if self.data_format == 'NCHW':
            image1 = tf.transpose(image1, [0, 3, 1, 2])
            image2 = tf.transpose(image2, [0, 3, 1, 2])

        # Pseudo Label
        logits2, _ = self.get_logits(image2)
        label2 = tf.nn.softmax(logits2)
        
        # Change this line if you modified training schedule or batchsize: 60 Epoch_num, 256 Batch_size
        k = tf.cast(get_global_step_var(), tf.float32) /  (60 * 1280000 / 256)
        
        # Sample lambda
        dist_beta = tf.distributions.Beta(1.0, 1.0)
        lmb = dist_beta.sample(tf.shape(image1)[0])
        lmb_x = tf.reshape(lmb, [-1, 1, 1, 1])
        lmb_y = tf.reshape(lmb, [-1, 1])
        
        # Interpolation        
        label_ori = label1
        if is_training:
            image = tf.to_float(image1) * lmb_x + tf.to_float(image2) * (1. - lmb_x)
            label = tf.stop_gradient(tf.to_float(tf.one_hot(label1, 1000)) * lmb_y + tf.to_float(label2) * (1. - lmb_y))
        else:
            image = image1
            label = tf.to_float(tf.one_hot(label1, 1000))

        # Calculate feats and logits for interpolated samples 
        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            logits, features = self.get_logits(image)
        
        # Classification Loss and error 
        loss = ImageNetModel.compute_loss_and_error(
            logits, label, label_smoothing=self.label_smoothing, lmb=lmb, label_ori=label_ori)

        # Distribution Alignment 
        lp = 2. / (1. + tf.exp(-10. * k)) - 1
        net_ = flip_gradient(features, lp)
        fc1 = FullyConnected('linear_1', net_, 1024, nl=tf.nn.relu)
        fc2 = FullyConnected('linear_2', fc1, 1024, nl=tf.nn.relu)
        domain_logits = FullyConnected("logits_dm", fc2, 2)
        label_dm = tf.concat([tf.reshape(lmb, [-1, 1]), tf.reshape(1. - lmb, [-1, 1])], axis=1)
        da_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label_dm, logits=domain_logits))
        
        # Final Loss
        loss += da_cost


        if self.weight_decay > 0:
            wd_loss = regularize_cost(self.weight_decay_pattern,
                                      tf.contrib.layers.l2_regularizer(self.weight_decay),
                                      name='l2_regularize_loss')
            add_moving_summary(loss, wd_loss)
            total_cost = tf.add_n([loss, wd_loss], name='cost')
        else:
            total_cost = tf.identity(loss, name='cost')
            add_moving_summary(total_cost)

        if self.loss_scale != 1.:
            logger.info("Scaling the total loss by {} ...".format(self.loss_scale))
            return total_cost * self.loss_scale
        else:
            return total_cost

    @abstractmethod
    def get_logits(self, image):
        """
        Args:
            image: 4D tensor of ``self.input_shape`` in ``self.data_format``

        Returns:
            Nx#class logits
        """

    def optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=0.1, trainable=False)
        tf.summary.scalar('learning_rate-summary', lr)
        return tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True)

    def image_preprocess(self, image):
        with tf.name_scope('image_preprocess'):
            if image.dtype.base_dtype != tf.float32:
                image = tf.cast(image, tf.float32)
            mean = [0.485, 0.456, 0.406]    # rgb
            std = [0.229, 0.224, 0.225]
            if self.image_bgr:
                mean = mean[::-1]
                std = std[::-1]
            image_mean = tf.constant(mean, dtype=tf.float32) * 255.
            image_std = tf.constant(std, dtype=tf.float32) * 255.
            image = (image - image_mean) / image_std
            return image

    @staticmethod
    def compute_loss_and_error(logits, label, label_smoothing=0., lmb=1.,label_ori=-1):
        loss = lmb * tf.losses.softmax_cross_entropy(
                label, logits, label_smoothing=label_smoothing)
        loss = tf.reduce_mean(loss, name='xentropy-loss')

        def prediction_incorrect(logits, label, topk=1, name='incorrect_vector'):
            with tf.name_scope('prediction_incorrect'):
                x = tf.logical_not(tf.nn.in_top_k(logits, label, topk))
            return tf.cast(x, tf.float32, name=name)

        wrong = prediction_incorrect(logits, label_ori, 1, name='wrong-top1')
        add_moving_summary(tf.reduce_mean(wrong, name='train-error-top1'))

        wrong = prediction_incorrect(logits, label_ori, 5, name='wrong-top5')
        add_moving_summary(tf.reduce_mean(wrong, name='train-error-top5'))
        return loss


if __name__ == '__main__':
    import argparse
    from tensorpack.dataflow import TestDataSpeed
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--aug', choices=['train', 'val'], default='val')
    args = parser.parse_args()

    if args.aug == 'val':
        augs = fbresnet_augmentor(False)
    elif args.aug == 'train':
        augs = fbresnet_augmentor(True)
    df = get_imagenet_dataflow(
        args.data, 'train', args.batch, augs)
    # For val augmentor, Should get >100 it/s (i.e. 3k im/s) here on a decent E5 server.
    TestDataSpeed(df).start()
