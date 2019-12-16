# -*- coding: utf-8 -*-
from __future__ import print_function


class NoneCompressor():
    @staticmethod
    def compress(tensor, name=None):
        return tensor, tensor.dtype

    @staticmethod
    def decompress(tensor, ctc, name=None):
        z = tensor 
        return z 


compressors = {
        'none': NoneCompressor,
        None: NoneCompressor
        }

