# code for smile
code for paper: One Positive Label is Sufficient: Single-Positive Multi-Label Learning with Label Enhancement(nips2022))

## Running environment

```bash
conda env create -f environment.yaml
```

## Run the Demo

```bash
python -u train_smile2.py --ep 10 --gpu ${gpu} --bias 0 --ds CAL500 --p 0.01 --mu 0.01 --rate 0.1
python -u train_smile2.py --ep 10 --gpu ${gpu} --bias 0 --role 0 --ds Image --p 0.1 --mu 0.9 --rate 0.7
python -u train_smile2.py --ep 10 --gpu ${gpu} --bias 0 --ds scene --p 0.075 --mu 0.3 --rate 0.5
python -u train_smile2.py --ep 10 --gpu ${gpu} --bias 0 --ds yeast --p 0.05 --mu 0.5 --rate 0.1
python -u train_smile2.py --ep 10 --gpu ${gpu} --bias 0 --ds corel5k --p 0.025 --mu 0.1 --rate 0.1
python -u train_smile2.py --ep 10 --gpu ${gpu} --bias 0 --ds rcv1subset1 --p 0.01 --mu 0.01 --rate 0.1
python -u train_smile2.py --ep 10 --gpu ${gpu} --bias 1 --ds Corel16k001 --p 0.001 --mu 0.01
python -u train_smile2.py --ep 10 --gpu ${gpu} --bias 1 --ds delicious --p 0.01 --mu 0.01
python -u train_smile2.py --ep 10 --gpu ${gpu} --bias 1 --ds iaprtc12 --p 0.001 --mu 0.01
python -u train_smile2.py --ep 10 --gpu ${gpu} --bias 1 --ds espgame --p 0.025 --mu 0.01
python -u train_smile2.py --ep 10 --gpu ${gpu} --bias 0 --ds mirflickr --p 0.01 --mu 0.01 --rate 0.1
python -u train_smile2.py --ep 10 --gpu ${gpu} --bias 0 --ds tmc2007 --p 0.025 --mu 0.1 --rate 0.3
```

# Datasets

The datasets are available at https://drive.google.com/drive/folders/1xre_GYxcn40Lj_qe5qLmuvfREW62eirP?usp=sharing.
