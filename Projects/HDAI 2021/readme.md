# H.D.A.I. 2021
# Team HPIC


 http://hpic.korea.ac.kr/
 - magental@korea.ac.kr
 - hanjungwoo1@korea.ac.kr







# Package Install
The latest codes are tested on Ubuntu 20.04, CUDA11.4, PyTorch 1.8 and Python 3.6:

CPU : AMD Ryzen 9 5950X 16-Core (32-Thread)

GPU : RTX 3090 (24GB)

```sh
$ pip install -r requirements.txt
```

# Segmentation (A2C / A4C)

### Data Preparation
Download [here](https://drive.google.com/file/d/12055x4BBl_Jwz02tiJnmQXMJwenr7uhV/view) and save in /data/echocardiography/.

### Run

You can run different modes with following codes.
- If you want to use A2C of data, you can use --A2C arguments
- If you want to use A4C of data, you can use --A4C arguments

```
## Select different models in ./model(default:UNET)
python train.py --task A2C --model other_model

## e.g., train Unet with A2C data
python train.py --task A2C
python test.py --task A2C

## e.g., train Unet with A4C data
python train.py --task A4C
python test.py --task A4C

## e.g., batchsize
python train.py --A2C --batch_size 2

```

### Performance


| A2C      | UNET   |
| -------- | ------ |
| DSC      | 0.9927 |
| Jacc     | 0.9855 |
| DSC_val  | 0.9864 |
| Jacc_val | 0.9733 |



| A4C      | UNET   |
| -------- | ------ |
| DSC      | 0.9931 |
| Jacc     | 0.9864 |
| DSC_val  | 0.9852 |
| Jacc_val | 0.9710 |



## License

MIT