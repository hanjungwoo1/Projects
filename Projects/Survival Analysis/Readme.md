# Survival Analysis

## Overview
 1. find Missing data
 2. Missing data imputation using generative adversarial nets
```bibtext
@inproceedings{yoon2018gain,
  title={Gain: Missing data imputation using generative adversarial nets},
  author={Yoon, Jinsung and Jordon, James and Schaar, Mihaela},
  booktitle={International conference on machine learning},
  pages={5689--5698},
  year={2018},
  organization={PMLR}
}
```
 3. Linear regression based classification
 4. Kaplan-Meier Estimator
 5. compare experiments(4 and 5)

## Code
 - Code_2020020592_한정우.py
 - End to End
 
## Data
 - ADNI Datasets
 
### Step 1.
 - Generator는 Missing 부분을 완전히 채우고, Discriminator의 오차를 최대화 하는게 목표
 - Discriminator는 imputed 값과 관측 값 사이의 분별, 분류 Loss를 최소화 하는게 목표
 - Hint Generator는 Discriminator의 학습 방향을 도와주는 역할

 Result : 처음에는 Loss가 떨어지지 않아 진행 도중에 어려움이 많았으나, 기존 Model을 참고하여 Tuning하는 과정에서 문제를 해결 하였음

### Step 2.
 - Keras의 Dense Layer를 사용하여 MLP으로 해결

 Result : 구현에 어려움은 없었으나, Step 1.의 데이터를 만들기에 오래 걸림 또한 MLP 모델을 구축함에 있어 Model Validation(Overfitting)이 불명확함

### Step 3.
 - Imputed된 Data, Step 1.에 Kaplan-Meier Estimation을 적용 

Kaplan-Meier Estimation : 생존 분석(Survival Analysis)에 적용 가능한 기법 중 하나, 시간의 흐름에 따른 어떠한 사건의 발생 확률을 알아보는 통계 분석 및 예측 기법.

### Step 4.
 - Step2의 경우 Input에 대한 Label을 확인하여 생존 가능성을 예측하기 좋음
 - Step3의 경우 각각의 Feature를 확인할 수 있고, 더 Survival Function에 영향을 많이주는 Feature를 파악하기 좋음
