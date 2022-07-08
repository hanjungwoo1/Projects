# Week #1 - Trainer Programming


# Experimental Setup

## Package Install
```sh
$ pip install -r src/requirements.txt
```

```bash
intern_data2022
├──train
├──valid
Now
├── src
│   ├── ___.py
│   ├── ___.py
│   ├── requirements.txt
│   ├── metric.py
│   └── dataloader.py
├── model
│   └── model.dat
├── Readme.md
└── run.sh
``` 

## Data EDA

### Data Preparation
Download [Box item] in wiki and save in ../intern_data2022/.

### read Data

데이터는 각각의 파일에 readline으로 읽은 뒤 Parsing하여 저장한다.

각 파일마다 저장하여 DataFrame으로 반환하는 형태로 만들었으며, 추후에 다른 기능들과 함께 Class해야 한다. 
 - dataloader.py



### Feature의 종류
 - gender : [0, 1, 2]
 - age_range : [U, -14, 15-19, 20-24, 25-29, 30-34, 35-39, 40-44, 45-49, 50-54, 55-59, 60-64]
 - os : ["IOS", "ANDROID"]
 - ad_id : ads number
 - inv_key : inventory_key
 - freq : Frequency

### Default Setting

- Label Encoding - [age_range, ad_id, inv_key]
- One Hot Encoding - [gender, OS]
- Numeric - freq

Label Encoding

  - 순서의 의미가 있을 때
  - 고유값의 개수가 많은데 One-hot Encoding을 사용할 수 없을 때

One Hot Encoding
 - 순서가 없을 때
 - 고유값의 개수가 많지 않을 때


#### event_id with bid_id
 - event_id(101)은 전환된 데이터, event_id(100)은 보여진 데이터
 - 6일치 데이터 특성상 전날에 보여진 데이터를 눌렀을 수 있음 -> 100은 없고, 101만 있는 상태
 - 같은 광고가 중복되어 보여 졌을 수 있음 -> event_id 와 bid_id 중복
 - 클릭된 광고(101)을 학습해야 하기에, 중복된 데이터를 제거하며, bid_id가 같은 100을 제거
   - Data를 event_id, bid_id를 기준으로 정렬
 ```python
train_df_sort = train_df.sort_value(by=['event_id', 'bid_id'], ascending=True)
```
   - bid_id와 중복된 데이터를 제거하며 마지막만 남겨둠 = 고유의 bid_id와 event_id(101)
```python 
   train_df_sort = train_df.drop_duplicates(["bid_id"], keep='last')
```

### preprocessing Experiments

#### sorted Data Problem - Shuffle

데이터를 정렬하여 사용하였기 때문에 학습에 문제가 일어나는가

```python
train_shuffle = train_data.sample(frac=1).reset_index(drop=True)
```
동일한 학습 결과를 나타냈다.

#### overSampling



# Model 

#### 모델 선택

각각의 모델 후보군이며 Name = Number of parameters / Order of feature interactions 으로 기재했다. 

1. Multivariate Statistical Models
-  Logistic Regression (LR) =  n / 1
-  Degree-2 Polynomial (Poly2) =  n + H / 2

2. Factorization Machines (FMs) based Models
- Factorization Machines (FMS) / n + nk / 2
  - Powerful Feature Interactions in FMs
- Field-aware Factorization Machines (FFMs) / n + n(m-1)k / 2
- Field-weighted Factorization Machines (FwFMs) / n + nk + m^2 / 2

Survey 논문에 기재된 정확도, LogLoss를 기준으로 성능의 순서는 이렇다.

LR < Poly2 < FM < FFMs < Others(DNN)

~~3. Deep Learning Models
-  Long Short-Term Memory (LSTM) / / >2
-  Convolutional Neural Network (CNN) / / >2
-  Factorization Machine supported Neural Network (FNN) / / >2
-  DeepFM / / >2~~

딥러닝 방식은 GPU를 사용하지 않아 후보군에서 제외하였다.
 - Todo : DeepFM의 Inference Time을 체크해봐야 한다.

4. Tree Models 
 - Gradient Boosting Decision Tree (GBDT) / / >2
 - XGBoost / / >2
 - CatBoost / / >2
 - LightGBM / / >2
 - 





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

|             | CatBoost_Regression | LightGBM    |
|-------------|---------------------|-------------|
| Calib       | 1.03495148          | 1.018517195 |
| RIG         | 0.07394825          | 0.080642409 |
| Training Time | 104.32s             | 59.15s      |
| Inference Time | 1.02s               | 9.41s       |



| A4C      | UNET   |
| -------- | ------ |
| DSC      | 0.9931 |
| Jacc     | 0.9864 |
| DSC_val  | 0.9852 |
| Jacc_val | 0.9710 |




### ToDo
 앞으로 해야할 일
 - ㅇㅇ 
 - ㅇㅇ
 - ㅇㅇ


Future Challenge
- Feature Engineering for CTR Prediction
- Sample Imbalance between Clicks and Non-clicks
- CTR Prediction for New Advertisements: Cold Start
- Data Sparsity and Heterogeneity for CTR Prediction




## License
in Line