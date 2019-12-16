For DeepSpeech


Pytorch 0.4.1
Cuda 9.0 cuDNN 7.4
Ubuntu 14.04
GCC 4.8.4

Install warp-ctc
export CUDA_HOME="/usr/local/cuda-9.0”
git clone https://github.com/SeanNaren/warp-ctc.git
cd warp-ctc
mkdir build; cd build
cmake ..
make
cd ../pytorch_binding
python setup.py install

install torchaudio
sudo apt-get install sox libsox-dev libsox-fmt-all
git clone https://github.com/pytorch/audio.git
cd audio
python setup.py install

pip install -r requirements.txt

install data
cd data; python an4.py
cd data; python librispeech.py
(data path: P100 /home/yxwang/proj/DGC_speech2text/data/ )

Run
An4:
python train.py  --rnn-type lstm --hidden-size 800 --hidden-layers 5
--checkpoint --train-manifest
/home/yxwang/proj/DGC_speech2text/data/an4_train_manifest.csv --val-manifest
/home/yxwang/proj/DGC_speech2text/data/an4_val_manifest.csv --epochs 100
--num-workers $(nproc) --cuda --batch-size 32 --learning-anneal 1.01 --augment
Librispeech:
python train.py  --rnn-type gru --hidden-size 1200 --hidden-layers 7
--checkpoint --train-manifest
/home/yxwang/proj/DGC_speech2text/data/libri_train_manifest.csv --val-manifest
/home/yxwang/proj/DGC_speech2text/data/libri_val_manifest.csv --epochs 15
--num-workers $(nproc) --cuda --batch-size 20 --learning-anneal 1.1

