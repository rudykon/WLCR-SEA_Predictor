# Limitations and appropriate use

Use the project for research and controlled tests. Validate it again before live deployment.

## Data and generalization

- One anonymous region, about one month.
- Every forecast starts at midnight; horizon and time-of-day effects overlap.
- An August test set influenced some design choices.
- Cell-disjoint tests still use the same regional trace.
- New regions, seasons, metrics, or collection systems need new tests.

## Accuracy interpretation

- DLinear has the lowest complete-data WAPE.
- WLCR-SEA and the prior method are not clearly different on complete data.
- Some moderate missing-rate comparisons are also unclear.
- Missing-data gains apply only to the tested patterns and runs.
- Routing entropy is not a reliable uncertainty estimate.

## Serving and privacy

Blocking cell-ID lookups does **not** provide:

- access control or authorization;
- encryption at rest or in transit;
- anonymization or differential privacy;
- a guarantee that logs, caches, data gateways, or downstream systems are safe.

Production still needs access control, retention, safe logs, encryption, and incident response.

## Public Demo boundary

The public Space runs `A0_fixed`, not the trained A6 model. Its fallback comes from the current input.

!!! danger "Demo ≠ A6"
    Do not compare Demo output with the reported A6 results.
