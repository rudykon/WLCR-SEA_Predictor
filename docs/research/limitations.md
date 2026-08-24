# Limitations and appropriate use

The available evidence supports research and controlled testing, but it is not enough to claim that the model will work unchanged in every region, season, or live network. The main limitations are listed below.

## Data and generalization

- The study covers one anonymous region for roughly one month.
- Every forecast starts at midnight. The study therefore cannot separate the effect of “hours ahead” from the effect of “time of day.”
- An August test set influenced some design choices, so the final findings should still be treated as exploratory.
- Tests with different training and test cells still use the same regional trace. They do not establish performance across regions, operators, or seasons.
- A new data-collection system, metric definition, or traffic pattern requires a new evaluation.

## Accuracy interpretation

- DLinear has the lowest complete-data WAPE in the reported comparison.
- WLCR-SEA and the prior traffic-only method do not show a clear difference on complete data because the confidence interval includes zero.
- At some moderate missing rates, differences from selected baselines are also unclear.
- The strongest missing-data results apply only to the fixed removal patterns and retraining runs used in the study.
- Routing entropy is not a reliable uncertainty estimate in this study.

## Serving and privacy

During a prediction, the model is prevented from using the cell ID to fetch additional traffic data. This restriction does **not** by itself provide:

- access control or authorization;
- encryption at rest or in transit;
- anonymization or differential privacy;
- a guarantee that logs, caches, data gateways, or downstream systems are safe.

The cell ID may still exist outside the model for access control and record keeping. A production system still needs clear rules for data retention, logging, encryption keys, and incident response.

## Public Demo boundary

The repository does not include the trained A6 checkpoint or the training-set fallback values. The public Space therefore runs the simpler `A0_fixed` baseline and calculates its final fallback from the current input. This is sufficient to demonstrate the calculation steps, but it is not the trained model used in the paper.

!!! danger "Do not compare Demo outputs with the paper tables"
    To reproduce the paper's results, use the documented dataset, data splits, training process, checkpoints, and evaluation scripts.
