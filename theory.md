# Mathematical Theory & Background

## [TO BE FILLED BY YOU]

This document should contain your thesis theory including:

### 1. Problem Formulation
- Objective function
- Decision variables
- Constraints

### 2. Market Microstructure Models
- Power law market impact (Curato et al. 2014)
- Bid-ask spread costs
- Inventory risk (Almgren-Chriss 2001)
- Permanent vs transient impact decomposition

### 3. Optimization Method
- Differential Evolution algorithm
- Why global optimization is needed
- Comparison with SQP and Dynamic Programming

### 4. Parameter Calibration
- How η (impact) is calibrated
- How γ (power law) is determined
- How σ (volatility) is measured
- Liquidity classification (ADV-based)

### 5. Validation Methods
- Perturbation testing
- Monte Carlo simulation
- Constraint satisfaction tests

### 6. Literature Review
- Almgren & Chriss (2001)
- Curato et al. (2014)
- Gatheral (2010)
- Other relevant papers

---

## Quick Reference: Cost Function

Total cost minimized by the solver:

```
Cost = Σᵢ [Impact(Sᵢ) + Spread(Sᵢ) + Risk(Xᵢ)]
```

Where:
- **Impact:** η |Sᵢ|^γ S₀ (power law with permanent + transient)
- **Spread:** Sᵢ × (spread/2) × S₀ (bid-ask spread)
- **Risk:** (λσ²/2) Xᵢ² Δt (Almgren-Chriss inventory risk)

---

## Code Location

All theory is implemented in: `solver.py`
- Lines 100-200: Cost models
- Lines 200-400: Optimization setup
- Lines 400-600: Differential Evolution solver

---

**Status:** Template - Fill this with your thesis content
