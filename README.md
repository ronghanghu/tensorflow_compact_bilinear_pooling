# Compact Bilinear Pooling

This repository contains the tensorflow implementation of Compact Bilinear
Pooling.

Details in `compact_bilinear_pooling_layer` in `compact_bilinear_pooling.py`.
```
def compact_bilinear_pooling_layer(bottom1, bottom2, output_dim, sum_pool=True,
    rand_h_1=None, rand_s_1=None, rand_h_2=None, rand_s_2=None,
    seed_h_1=1, seed_s_1=3, seed_h_2=5, seed_s_2=7, sequential=True,
    compute_size=128)
```

Reference:

    Yang Gao, et al. "Compact Bilinear Pooling." in Proceedings of IEEE
    Conference on Computer Vision and Pattern Recognition (2016).
    Akira Fukui, et al. "Multimodal Compact Bilinear Pooling for Visual Question
    Answering and Visual Grounding." arXiv preprint arXiv:1606.01847 (2016).
