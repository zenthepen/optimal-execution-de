"""
Simple Example: Optimal Execution Solver

This example shows how to use the solver in 3 simple steps.
"""

import sys
import numpy as np

# Import the solver
from solver import OptimalExecutionRealistic


def main():
    """Run a basic optimal execution example"""
    
    print("=" * 60)
    print("OPTIMAL EXECUTION SOLVER - EXAMPLE")
    print("=" * 60)
    print()
    
    # Step 1: Create solver with SNAP stock parameters
    print("Step 1: Initializing solver...")
    print("  - Order size: 100,000 shares")
    print("  - Time horizon: 1 day (10 periods)")
    print("  - Stock: SNAP ($10/share)")
    print()
    
    solver = OptimalExecutionRealistic(
        X0=100000,              # Order size: 100,000 shares
        T=1.0,                  # Time horizon: 1 day
        N=10,                   # Number of periods
        eta=0.035,              # Impact coefficient (calibrated for SNAP)
        lam=1e-6,               # Risk aversion
        sigma=0.02,             # Volatility (2% daily)
        gamma=0.67,             # Power law exponent
        S0=10.0                 # Initial stock price: $10
    )
    
    # Step 2: Solve for optimal strategy
    print("Step 2: Solving optimization problem...")
    print("  (Using Differential Evolution global optimizer)")
    print()
    
    result = solver.solve()
    
    # Step 3: Display results
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print()
    
    print(f"✅ Status: {result['status']}")
    print(f"   Iterations: {result['iterations']}")
    print(f"   Function Evaluations: {result['function_evals']}")
    print()
    
    print(f"💰 COSTS")
    print(f"   Total Cost:  ${result['total_cost']:.2f}")
    print(f"   - Impact:    ${result['impact_cost']:.2f} ({result['impact_cost']/result['total_cost']*100:.1f}%)")
    print(f"   - Spread:    ${result['spread_cost']:.2f} ({result['spread_cost']/result['total_cost']*100:.1f}%)")
    print(f"   - Risk:      ${result['risk_cost']:.2f} ({result['risk_cost']/result['total_cost']*100:.1f}%)")
    print()
    
    print(f"📊 PERFORMANCE")
    print(f"   Improvement vs TWAP: {result['improvement_vs_twap']:.2f}%")
    print()
    
    print("📈 OPTIMAL EXECUTION SCHEDULE")
    print("   Period | Shares     | % of Total | Remaining")
    print("   " + "-" * 50)
    
    remaining = 100000
    for i, shares in enumerate(result['optimal_strategy'], 1):
        pct = (shares / 100000) * 100
        print(f"   {i:6d} | {shares:10,.0f} | {pct:10.2f}% | {remaining:9,.0f}")
        remaining -= shares
    
    print()
    print("=" * 60)
    print("KEY INSIGHTS")
    print("=" * 60)
    print()
    print("✓ Front-loading: Trade more aggressively early to reduce risk")
    print("✓ Gradual taper: Reduce trade sizes as inventory decreases")
    print("✓ Constraint-aware: Respects 20% max per period liquidity limit")
    print("✓ Cost-optimal: Balances impact, spread, and risk costs")
    print()
    print("=" * 60)
    
    return result


if __name__ == "__main__":
    try:
        result = main()
        print("\n✅ Example completed successfully!")
        print("\nNext steps:")
        print("  - Run tests: python tests.py")
        print("  - View results: ls results/")
        print("  - Read theory: cat theory.md")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure you have installed all dependencies:")
        print("  pip install numpy scipy matplotlib pandas yfinance")
        sys.exit(1)
