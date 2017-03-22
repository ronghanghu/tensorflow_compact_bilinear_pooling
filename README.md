# Compact Bilinear Pooling

This repository contains the tensorflow implementation of Compact Bilinear Pooling.

## Usage

Details of this operation can be seen in `compact_bilinear_pooling_layer` in `compact_bilinear_pooling.py`.
```
def compact_bilinear_pooling_layer(bottom1, bottom2, output_dim, sum_pool=True,
    rand_h_1=None, rand_s_1=None, rand_h_2=None, rand_s_2=None,
    seed_h_1=1, seed_s_1=3, seed_h_2=5, seed_s_2=7, sequential=True,
    compute_size=128)
    """
    Compute compact bilinear pooling over two bottom inputs. Reference:
    Yang Gao, et al. "Compact Bilinear Pooling." in Proceedings of IEEE
    Conference on Computer Vision and Pattern Recognition (2016).
    Akira Fukui, et al. "Multimodal Compact Bilinear Pooling for Visual Question
    Answering and Visual Grounding." arXiv preprint arXiv:1606.01847 (2016).
    Args:
        bottom1: 1st input, 4D Tensor of shape [batch_size, height, width, input_dim1].
        bottom2: 2nd input, 4D Tensor of shape [batch_size, height, width, input_dim2].
        output_dim: output dimension for compact bilinear pooling.
        sum_pool: (Optional) If True, sum the output along height and width
                  dimensions and return output shape [batch_size, output_dim].
                  Otherwise return [batch_size, height, width, output_dim].
                  Default: True.
        rand_h_1: (Optional) an 1D numpy array containing indices in interval
                  `[0, output_dim)`. Automatically generated from `seed_h_1`
                  if is None.
        rand_s_1: (Optional) an 1D numpy array of 1 and -1, having the same shape
                  as `rand_h_1`. Automatically generated from `seed_s_1` if is
                  None.
        rand_h_2: (Optional) an 1D numpy array containing indices in interval
                  `[0, output_dim)`. Automatically generated from `seed_h_2`
                  if is None.
        rand_s_2: (Optional) an 1D numpy array of 1 and -1, having the same shape
                  as `rand_h_2`. Automatically generated from `seed_s_2` if is
                  None.
        sequential: (Optional) if True, use the sequential FFT and IFFT
                    instead of tf.batch_fft or tf.batch_ifft to avoid
                    out-of-memory (OOM) error.
                    Note: sequential FFT and IFFT are only available on GPU
                    Default: True.
        compute_size: (Optional) The maximum size of sub-batch to be forwarded
                      through FFT or IFFT in one time. Large compute_size may
                      be faster but can cause OOM and FFT failure. This
                      parameter is only effective when sequential == True.
                      Default: 128.
    Returns:
        Compact bilinear pooled results of shape [batch_size, output_dim] or
        [batch_size, height, width, output_dim], depending on `sum_pool`.
    """
```

## Testing

To test whether it works correctly on your system, run:
```
python compact_bilinear_pooling_test.py
```
The tests pass if no error occurs running the above command.

Note that `sequential=True` (Default) only supports GPU computation, with no CPU kernel available.

# Building

The `sequential_fft/build/sequential_batch_fft.so` is built against TensorFlow
version 1.0.0 with CUDA 8.0 and g++ 4.8.4, which should be compatible with the
official build of TensorFlow 1.0.0 on Ubuntu/Linux 64-bit.

If you set `sequential=True` (Default), you will need this `sequential_batch_fft.so` to be compatible with your TensorFlow installation.

If installed TensorFlow from source, or want to use a different version of TensorFlow
other than 1.0.0 that may be built with a different compiler and a different CUDA
version, you may need to rebuild `sequential_batch_fft.so` with `compile.sh` in `sequential_fft/`,
*using the same CUDA version and C++ compiler*. To see the compiler version of an official TF build,
run in Python the follows.
```
import tensorflow as tf; print(tf.__compiler_version__)
```

## Reference

    Yang Gao, et al. "Compact Bilinear Pooling." in Proceedings of IEEE
    Conference on Computer Vision and Pattern Recognition (2016).
    Akira Fukui, et al. "Multimodal Compact Bilinear Pooling for Visual Question
    Answering and Visual Grounding." arXiv preprint arXiv:1606.01847 (2016).
