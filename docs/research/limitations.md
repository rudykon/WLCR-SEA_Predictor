# Limitations and appropriate use

The current evidence supports research and controlled testing. It is not enough
to claim that the model will work unchanged in every region, season, or live
network. The main limits are listed below.

## Data and generalization

- The study covers one anonymous region for roughly one month.
- Every forecast starts at midnight. The study therefore cannot separate the
  effect of “hours ahead” from the effect of “time of day.”
- An August test set influenced some design choices, so the final findings
  should still be treated as exploratory.
- Tests with different training and test cells still use the same regional
  trace. They do not prove performance across regions, operators, or seasons.
- A new data collection system, metric definition, or traffic pattern requires
  new evaluation.

## Accuracy interpretation

- DLinear has the lowest complete-data WAPE in the reported comparison.
- WLCR-SEA and the prior traffic-only method do not show a clear difference on
  complete data because the confidence interval includes zero.
- At some moderate missing rates, differences from selected baselines are also
  unclear.
- The strongest missing-data results apply to the fixed removal patterns and
  retraining runs used in the paper.
- Routing entropy is not a reliable uncertainty estimate in this study.

## Serving and privacy

The model is prevented from fetching extra traffic based on the cell ID during
one prediction. This restriction does **not** by itself provide:

- access control or authorization;
- encryption at rest or in transit;
- anonymization or differential privacy;
- a guarantee that logs, caches, data gateways, or downstream systems are safe.

The cell ID can still exist outside the model for access control and record
keeping. A production system still needs clear rules for data retention, logs,
encryption keys, and incident response.

## Public Demo boundary

The repository does not include the trained A6 checkpoint or the training-set
fallback values. The public Space therefore runs the simpler `A0_fixed`
baseline and uses a value calculated from the current input as its final
fallback. This is enough to demonstrate the calculation steps, but it is not
the paper's trained model.

!!! danger "Do not compare Demo outputs with the paper tables"
    To reproduce the paper, use the documented dataset, splits, training
    process, checkpoints, and evaluation scripts.
