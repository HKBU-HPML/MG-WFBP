# -*- coding: utf-8 -*-
from __future__ import print_function

import logging
import socket

DEBUG =0

GPU_CONSTRUCTION=True
WARMUP=True
DELAY_COMM=1

PREFIX=''
if WARMUP:
    PREFIX=PREFIX+'gwarmup'

PREFIX=PREFIX+'-dc'+str(DELAY_COMM)
EXCHANGE_MODE = 'MODEL' 

PREFIX=PREFIX+'-'+EXCHANGE_MODE.lower()

CONNECTION='10GbE'
#CONNECTION='56GbIB'
EXP='-tpds'+CONNECTION
FP16=False
ADAPTIVE_MERGE=True
if FP16:
    EXP=EXP+'-fp16'
PREFIX=PREFIX+EXP
if ADAPTIVE_MERGE:
    PREFIX=PREFIX+'-ada'

FAKE_DATA=False
ORIGINAL_HOROVOD=False
if ORIGINAL_HOROVOD:
    PREFIX=PREFIX+'-hvd'

TENSORBOARD=False

MAX_EPOCHS = 200

hostname = socket.gethostname() 
logger = logging.getLogger(hostname)

if DEBUG:
    logger.setLevel(logging.DEBUG)
else:
    logger.setLevel(logging.INFO)

strhdlr = logging.StreamHandler()
logger.addHandler(strhdlr)
formatter = logging.Formatter('%(asctime)s [%(filename)s:%(lineno)d] %(levelname)s %(message)s')
strhdlr.setFormatter(formatter)

