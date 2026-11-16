## F1 by condition (mean ± std over seeds)

| experiment            | Exact-match rules   | Logistic Regression   | Random Forest   | XGBoost       | LightGBM      |
|:----------------------|:--------------------|:----------------------|:----------------|:--------------|:--------------|
| E0 clean              | 0.986 ± 0.004       | 1.000 ± 0.000         | 1.000 ± 0.000   | 1.000 ± 0.000 | 1.000 ± 0.000 |
| E1 10% ids missing    | 0.986 ± 0.004       | 0.993 ± 0.001         | 1.000 ± 0.000   | 1.000 ± 0.000 | 1.000 ± 0.000 |
| E2 30% ids missing    | 0.986 ± 0.004       | 0.988 ± 0.002         | 0.998 ± 0.001   | 0.997 ± 0.001 | 0.998 ± 0.000 |
| E3 E2 + amount noise  | 0.957 ± 0.015       | 0.983 ± 0.003         | 0.997 ± 0.003   | 0.997 ± 0.002 | 0.999 ± 0.001 |
| E4 E2 + name variants | 0.938 ± 0.008       | 0.960 ± 0.007         | 0.993 ± 0.001   | 0.996 ± 0.001 | 0.997 ± 0.001 |
| E5 E2 + date shifts   | 0.967 ± 0.003       | 0.988 ± 0.001         | 0.997 ± 0.002   | 0.998 ± 0.001 | 0.999 ± 0.001 |
| E6 everything         | 0.835 ± 0.019       | 0.886 ± 0.023         | 0.987 ± 0.006   | 0.989 ± 0.004 | 0.992 ± 0.003 |

## Recall on true matches without identifier

| experiment            |   Exact-match rules |   Logistic Regression |   Random Forest |   XGBoost |   LightGBM |
|:----------------------|--------------------:|----------------------:|----------------:|----------:|-----------:|
| E0 clean              |             nan     |                   nan |         nan     |   nan     |    nan     |
| E1 10% ids missing    |               1     |                     1 |           1     |     1     |      1     |
| E2 30% ids missing    |               1     |                     1 |           0.995 |     0.996 |      1     |
| E3 E2 + amount noise  |               0.794 |                     1 |           0.992 |     0.997 |      1     |
| E4 E2 + name variants |               0.686 |                     1 |           0.991 |     0.992 |      0.998 |
| E5 E2 + date shifts   |               0.851 |                     1 |           0.989 |     0.997 |      1     |
| E6 everything         |               0.485 |                     1 |           0.994 |     0.992 |      0.994 |

## LightGBM F1 by training protocol

| experiment            |   Trained on E2 only |   Trained on all conditions |   Trained on the same condition |
|:----------------------|---------------------:|----------------------------:|--------------------------------:|
| E0 clean              |                1     |                       1     |                           1     |
| E1 10% ids missing    |                1     |                       0.999 |                           1     |
| E2 30% ids missing    |                0.998 |                       0.998 |                           0.998 |
| E3 E2 + amount noise  |                0.941 |                       0.999 |                           0.999 |
| E4 E2 + name variants |                0.891 |                       0.997 |                           0.997 |
| E5 E2 + date shifts   |                0.999 |                       0.999 |                           0.999 |
| E6 everything         |                0.802 |                       0.992 |                           0.992 |

## Ablation (E6)

|                   |   f1_without |   pr_auc_without |   f1_alone |   pr_auc_alone |   f1_drop |
|:------------------|-------------:|-----------------:|-----------:|---------------:|----------:|
| All features      |       0.9956 |           0.9986 |     0.9956 |         0.9986 |    0      |
| Identifier        |       0.9941 |           0.9945 |     0.6751 |         0.5281 |    0.0015 |
| Amount            |       0.9723 |           0.9878 |     0      |         0.1273 |    0.0232 |
| Dates             |       0.9904 |           0.9968 |     0      |         0.1841 |    0.0052 |
| Rate              |       0.9904 |           0.995  |     0      |         0.2295 |    0.0052 |
| Counterparty      |       0.9773 |           0.9928 |     0.0175 |         0.2075 |    0.0183 |
| Direction/product |       0.9904 |           0.9952 |     0      |         0.0746 |    0.0052 |

## SHAP share by group

|                       |   Identifier |   Amount |   Dates |   Rate |   Counterparty |   Direction/product |
|:----------------------|-------------:|---------:|--------:|-------:|---------------:|--------------------:|
| E0 clean              |        0.936 |    0     |   0     |  0.064 |          0     |               0     |
| E1 10% ids missing    |        0.354 |    0.141 |   0.125 |  0.131 |          0.167 |               0.084 |
| E2 30% ids missing    |        0.109 |    0.182 |   0.205 |  0.187 |          0.203 |               0.114 |
| E3 E2 + amount noise  |        0.179 |    0.13  |   0.218 |  0.188 |          0.195 |               0.09  |
| E4 E2 + name variants |        0.24  |    0.186 |   0.202 |  0.21  |          0.089 |               0.073 |
| E5 E2 + date shifts   |        0.138 |    0.178 |   0.192 |  0.214 |          0.201 |               0.078 |
| E6 everything         |        0.156 |    0.224 |   0.253 |  0.139 |          0.116 |               0.113 |

## Threshold policies (E6 test)

|                           |   precision |   recall |     f1 |   pr_auc |   threshold |   cost |   false_positives |   false_negatives |
|:--------------------------|------------:|---------:|-------:|---------:|------------:|-------:|------------------:|------------------:|
| Default threshold         |      0.9832 |   0.9972 | 0.9901 |   0.9977 |        0.5  |   1830 |                36 |                 6 |
| Best F1 on validation     |      0.9859 |   0.9939 | 0.9899 |   0.9977 |        0.86 |   1565 |                30 |                13 |
| Lowest cost on validation |      0.9859 |   0.9939 | 0.9899 |   0.9977 |        0.86 |   1565 |                30 |                13 |