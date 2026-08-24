# Scope and limitations

The repository is strongest when its evidence boundary is kept explicit. These
limitations are part of the result, not footnotes to remove from deployment
materials.

## Data and generalization

- The study covers one anonymous region for roughly one month.
- All forecast origins are at midnight, so horizon and clock hour are confounded.
- The August holdout informed redesign decisions; the final evidence remains exploratory.
- Cell-disjoint refits separate cell identities inside the same trace. They do
  not establish cross-region, cross-operator, cross-season, or prospective generalization.
- Performance under a new telemetry process, indicator definition, or load
  regime must be measured rather than inferred from the reported intervals.

## Accuracy interpretation

- DLinear has the lowest clean WAPE in the reported comparison.
- The clean paired interval between WLCR-SEA and the prior traffic-only method
  includes zero; the study does not detect a difference there.
- Moderate missingness intervals against selected augmentation-matched baselines
  can include zero.
- Strong structured-missingness findings are conditional on fixed masks and refits.
- Routing entropy is not calibrated uncertainty in the reported evidence.

## Serving and privacy

Request-local scoring prevents the predictor from issuing identity-conditioned
traffic lookups during one forecast. It does not, by itself, provide:

- access control or authorization;
- encryption at rest or in transit;
- anonymization or differential privacy;
- a guarantee that logs, caches, ingress services, or downstream consumers are safe.

The cell identifier can still exist outside the forecasting function for
authorization, routing, and audit. Production systems need explicit retention,
logging, key-management, and incident-response policies.

## Public Demo boundary

The repository does not publish the fitted A6 checkpoint or frozen training
prior. The public Space therefore runs `A0_fixed`, the real parameter-free
ablation, and substitutes a request-derived value only for the last-resort
fallback slot. This makes the evidence mechanics executable without claiming
that the output is the paper model.

!!! danger "Do not benchmark the paper with the Space"
    Demo forecasts are not eligible for comparison with the manuscript tables.
    Reproduce the fitted experiments with the documented data, splits,
    checkpoints, and evaluation scripts instead.
