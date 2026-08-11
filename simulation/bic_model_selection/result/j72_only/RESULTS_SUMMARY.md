# Big model selection: n=1000 and J=72 only

The filtered analysis contains 7,500 candidate fits: three true-dimension
scenarios, 100 replicates per scenario, and 25 fitted `(K1,K2)` pairs per
replicate.

| True dimensions | Correct selections | Minimum BIC margin | Median BIC margin | Maximum BIC margin |
|---|---:|---:|---:|---:|
| (2,2) | 100/100 | 229.0 | 358.3 | 554.5 |
| (3,3) | 100/100 | 298.5 | 371.9 | 1077.6 |
| (4,4) | 100/100 | 293.4 | 394.1 | 635.3 |

The BIC margin is defined as the BIC of the best incorrect candidate minus the
BIC of the true candidate. Therefore, positive values favor the true model.
Every one of the 300 margins is positive, and even the smallest margin is
228.95. Thus, the 100% selection rates are not caused by near ties.

The closest incorrect alternatives usually differ from the truth by one latent
dimension. For true `(2,2)`, the best incorrect model is `(3,2)` in 85% of
replicates and `(2,3)` in 15%. For true `(3,3)`, it is `(4,3)` in 94%. For true
`(4,4)`, it is `(5,4)` in 87%, followed by `(3,4)` in 10%.

Only these J=72 results should be used in the manuscript table. The J=48
scenarios are intentionally excluded from every file in this folder.
