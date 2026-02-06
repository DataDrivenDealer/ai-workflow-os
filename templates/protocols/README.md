# Protocol Templates

This directory contains reusable protocol templates for the Research Protocol Algebra (RPA).
Use these templates as starting points when designing new experiments.

## Available Templates

| Template | Use Case | Key Features |
|----------|----------|--------------|
| `factor_development.yaml` | New factor research | IS/OOS split, MTC correction |
| `model_comparison.yaml` | Compare multiple models | Walk-forward, cross-validation |
| `hyperparameter_search.yaml` | Parameter optimization | Grid/random search, validation |
| `robustness_test.yaml` | Validate existing factor | Multiple windows, stress tests |
| `data_validation.yaml` | Data quality checks | Schema, coverage, consistency |

## Usage

1. Copy the template to your experiment directory:
   ```bash
   cp templates/protocols/factor_development.yaml experiments/t01_my_factor/config.yaml
   ```

2. Customize the parameters for your use case.

3. Run the protocol design skill:
   ```
   /dgsf_protocol_design
   ```

## Template Structure

Each template follows the RPA schema:

```yaml
protocol:
  name: "Protocol Name"
  template: "template_id"
  version: "1.0.0"
  
  # Data configuration
  data:
    source: "..."
    date_range: {...}
    universe: "..."
  
  # Split configuration
  splits:
    is_ratio: 0.6
    validation_ratio: 0.2
    oos_ratio: 0.2
    purge_days: 5
  
  # Metrics and thresholds
  metrics:
    primary: [...]
    secondary: [...]
  
  thresholds:
    pass: {...}
    fail: {...}
  
  # Execution steps
  steps:
    - name: "Step 1"
      action: "..."
```

## Adding New Templates

1. Create a new YAML file in this directory
2. Follow the existing structure
3. Document in this README
4. Reference in `configs/research_protocol_algebra.yaml`
