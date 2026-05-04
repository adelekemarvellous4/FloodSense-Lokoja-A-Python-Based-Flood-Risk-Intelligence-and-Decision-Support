# Model Design

The first FloodSense susceptibility model uses a transparent weighted overlay.

## Factors

- Elevation: lower terrain increases flood susceptibility.
- Slope: flatter terrain increases flood susceptibility.
- Flow accumulation: higher accumulation increases flood susceptibility.
- Distance to river or stream: shorter distance increases flood susceptibility.
- Rainfall: higher rainfall increases flood susceptibility.
- Land cover: built-up and water-related classes may increase susceptibility when reliable class codes are available.

## Default Weights

- Elevation: 0.20
- Slope: 0.15
- Flow accumulation: 0.20
- Distance to river/stream: 0.25
- Rainfall: 0.10
- Land cover: 0.10

Scores are normalized to 0-1 and classified into five risk classes: Very Low, Low, Moderate, High, and Very High.
