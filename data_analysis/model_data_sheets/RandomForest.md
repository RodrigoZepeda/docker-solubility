Random Forest
================

## Model Description

Here goes the model explanation

## Model metrics

Several models were fitted to asses convergence. R² was used as the
model’s metric.

![](RandomForest_files/figure-gfm/unnamed-chunk-1-1.png)<!-- -->

The following table compares the errors in both the training and the
validation sets for the model with 2^20
estimators.

|         | Train errors | Absolute Train errors | Validation errors | Absolute Validation errors |
| ------- | -----------: | --------------------: | ----------------: | -------------------------: |
| Min.    |  \-1.9845144 |             0.0001528 |       \-2.5650415 |                  0.0033202 |
| 1st Qu. |  \-0.1174892 |             0.0564283 |       \-0.3420936 |                  0.1382948 |
| Median  |    0.0155242 |             0.1205659 |         0.0060174 |                  0.2749576 |
| Mean    |  \-0.0058099 |             0.1642896 |       \-0.1048953 |                  0.4296114 |
| 3rd Qu. |    0.1229192 |             0.2158671 |         0.2536160 |                  0.5476571 |
| Max.    |    1.4323121 |             1.9845144 |         1.4257689 |                  2.5650415 |

## Example

You can run the model via the following
    command:

    docker run --rm -v ~/PATH/TO_FILE/YOU_WANT_TO_WORK_ON/:/data docker-solubility RandomForest

## See also

  - [User’s
    manual](https://github.com/RodrigoZepeda/docker-solubility/blob/master/Manual.md)
  - [README](https://github.com/RodrigoZepeda/docker-solubility/blob/master/README.md)
