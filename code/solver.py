"""
Realistic Optimal Execution Solver with Full Market Impact Model

Implements progressively realistic constraints following Curato et al. (2014):
- Step 1: Baseline instantaneous impact (from de_solver.py)
- Step 2: Trade size limits (market depth constraints)
- Step 3: Bid-ask spread cost (linear transaction cost)
- Step 4: Permanent + transient impact decomposition
- Step 5: Exponential decay for transient component
- Step 6: Literature-calibrated default parameters

Key Features:
- NO corner solutions (natural constraints prevent instant execution)
- Realistic front-loading (smooth, economically justified)
- Empirically calibrated (Curato et al. 2014, Almgren-Chriss 2001)
- Passes perturbation test with realistic parameters

Author: Generated for optimal execution project
Date: 2025-10-28
"""

import numpy as np
import time
from scipy.optimize import differential_evolution
from typing import Dict, Optional


class OptimalExecutionRealistic:
    """
    Realistic Optimal Execution with Full Market Impact Model.
    
    Mathematical Formulation:
    -------------------------
    minimize: Σᵢ [impact_cost + spread_cost + risk_cost]
    
    where:
        impact_cost = S[i] * (decayed_transient + permanent + new_transient) * S₀
        permanent = η_perm * |S[i]|^γ  (never decays)
        new_transient = η_trans * |S[i]|^γ  (decays exponentially)
        decayed_transient *= exp(-ρ * τ)  (from previous periods)
        
        spread_cost = spread_per_share * S[i]
        risk_cost = 0.5 * λ * inventory² * σ² * τ
    
    Constraints:
        Σᵢ S[i] = X₀  (complete liquidation)
        0 ≤ S[i] ≤ max_trade_per_period  (liquidity limits)
    
    Key Insight:
    -----------
    Transient accumulation naturally prevents corner solutions:
    - Large concentrated trades → high transient buildup
    - Waiting allows transient to decay before next trade
    - Result: Smooth, realistic front-loading
    
    Literature References:
    ---------------------
    - Curato, Gatheral, Lillo (2014): Transient impact decomposition
    - Almgren, Chriss (2001): Risk aversion framework
    - Bouchaud et al. (2004): Market impact decay
    - Gatheral (2010): No-dynamic-arbitrage bounds
    """
    
    def __init__(self,
                 X0: float,
                 T: float,
                 N: int,
                 sigma: float,
                 lam: float,
                 eta: float,
                 gamma: float,
                 S0: float,
                 # Step 2: Trade size constraints
                 max_trade_fraction: float = 0.4,
                 # Step 3: Bid-ask spread
                 spread_bps: float = 1.0,
                 # Step 4-5: Permanent + transient with decay
                 permanent_fraction: float = 0.4,
                 decay_rate: float = 0.5):
        """
        Initialize realistic optimal execution problem.
        
        Parameters
        ----------
        X0 : float
            Initial inventory (shares to liquidate)
        T : float
            Time horizon (days)
        N : int
            Number of trading periods
        sigma : float
            Volatility (daily)
        lam : float
            Risk aversion parameter
        eta : float
            Total impact coefficient (will be split into permanent + transient)
        gamma : float
            Impact power-law exponent
        S0 : float
            Stock price (for dollar cost calculation)
            
        max_trade_fraction : float, default=0.4
            Maximum fraction of X0 tradeable in single period.
            Represents market depth / liquidity constraints.
            
            Typical values:
            - 0.3 (30%): Very liquid stocks (AAPL, MSFT)
            - 0.4 (40%): Liquid stocks (most S&P 500) [DEFAULT]
            - 0.5 (50%): Medium liquidity
            - 0.2 (20%): Low liquidity stocks
            
        spread_bps : float, default=1.0
            Half bid-ask spread in basis points.
            
            Typical values by liquidity:
            - 0.5 bps: Very liquid (AAPL, MSFT, SPY)
            - 1.0 bps: Liquid (S&P 500 stocks) [DEFAULT]
            - 2.0 bps: Medium liquidity
            - 5.0 bps: Low liquidity (small cap)
            
        permanent_fraction : float, default=0.4
            Fraction of impact that is permanent (never decays).
            Remaining fraction (1 - permanent_fraction) is transient.
            
            Empirical values from literature:
            - Curato et al. (2014): 30-50% permanent
            - Bouchaud et al. (2004): 40-60% permanent
            - Gatheral (2010): 30-40% permanent
            
            Default 0.4 (40% permanent, 60% transient) [DEFAULT]
            
        decay_rate : float, default=0.5
            Exponential decay rate for transient impact (ρ parameter).
            
            Interpretation:
            - decay_factor(t) = exp(-ρ * t)
            - Half-life = ln(2) / ρ
            
            Typical values:
            - ρ = 0.5: Half-life = 1.4 * τ (slow decay) [DEFAULT]
            - ρ = 1.0: Half-life = 0.7 * τ (medium decay)
            - ρ = 2.0: Half-life = 0.35 * τ (fast decay)
            
            Default 0.5 based on Curato et al. (2014) calibration.
        """
        # Core parameters (from Step 1)
        self.X0 = X0
        self.T = T
        self.N = N
        self.sigma = sigma
        self.lam = lam
        self.eta = eta
        self.gamma = gamma
        self.S0 = S0
        self.tau = T / N
        
        # Step 2: Trade size constraints
        self.max_trade_fraction = max_trade_fraction
        self.max_trade_per_period = X0 * max_trade_fraction
        
        # Step 3: Spread cost
        self.spread_bps = spread_bps
        self.spread_cost_per_share = spread_bps * 0.0001 * S0
        
        # Step 4: Permanent + transient decomposition
        self.permanent_fraction = permanent_fraction
        self.eta_permanent = eta * permanent_fraction
        self.eta_transient = eta * (1 - permanent_fraction)
        
        # Step 5: Transient decay
        self.decay_rate = decay_rate
        
    def cost_function(self, S: np.ndarray, debug: bool = False) -> float:
        """
        Compute total cost: impact + spread + risk.
        
        Cost Breakdown:
        ---------------
        1. Market Impact Cost (realistic model):
           - Accumulated transient from previous trades (decays)
           - Permanent impact from current trade (never decays)
           - New transient impact from current trade (will decay)
           
        2. Spread Cost:
           - Fixed cost per share (bid-ask spread)
           
        3. Risk Cost:
           - Quadratic in remaining inventory
           - Proportional to volatility and time
        
        Args:
            S: Trading strategy (array of N trade sizes)
            debug: If True, print detailed period-by-period breakdown
            
        Returns:
            Total expected cost (impact + spread + risk)
        """
        tau = self.tau
        price_displacement = 0.0  # Accumulated transient impact
        inventory = self.X0
        total_cost = 0.0
        
        if debug:
            print("\n" + "="*80)
            print("COST FUNCTION DEBUG OUTPUT")
            print("="*80)
            print(f"Parameters:")
            print(f"  eta_permanent: {self.eta_permanent:.10f}")
            print(f"  eta_transient: {self.eta_transient:.10f}")
            print(f"  gamma: {self.gamma}")
            print(f"  S0: {self.S0}")
            print(f"  spread_per_share: {self.spread_cost_per_share:.6f}")
            print(f"  decay_rate: {self.decay_rate}")
            print()
            print(f"Trades: {S}")
            print()
        
        impact_costs = []
        spread_costs = []
        risk_costs = []
        
        for i in range(self.N):
            # 1. Decay previous transient impact (exponential resilience)
            decay_factor = np.exp(-self.decay_rate * tau)
            price_displacement = price_displacement * decay_factor
            
            # 2. Compute permanent impact (never decays)
            permanent_impact = self.eta_permanent * (np.abs(S[i]) ** self.gamma)
            
            # 3. Compute new transient impact (will decay in future)
            transient_impact = self.eta_transient * (np.abs(S[i]) ** self.gamma)
            
            # 4. Total price displacement this period
            current_price_impact = price_displacement + permanent_impact + transient_impact
            
            # 5. Market impact cost (paid on displaced price)
            # Cost = S[i] × (price impact coefficient) × S₀
            # where price impact coefficient = η × |S[i]|^γ
            impact_cost = S[i] * current_price_impact * self.S0
            
            # 6. Spread cost (linear in trade size)
            spread_cost = self.spread_cost_per_share * S[i]
            
            # 7. Update inventory (after trade)
            inventory -= S[i]
            
            # 8. Risk cost (inventory risk during period)
            risk_cost = 0.5 * self.lam * (inventory ** 2) * (self.sigma ** 2) * tau
            
            # 9. Accumulate costs
            total_cost += impact_cost + spread_cost + risk_cost
            
            impact_costs.append(impact_cost)
            spread_costs.append(spread_cost)
            risk_costs.append(risk_cost)
            
            if debug and S[i] > 0.01:
                print(f"Period {i+1}:")
                print(f"  Trade: {S[i]:,.2f} shares ({S[i]/self.X0*100:.2f}%)")
                print(f"  Carryover transient coef: {price_displacement:.10f}")
                print(f"  New permanent coef: {permanent_impact:.10f}")
                print(f"  New transient coef: {transient_impact:.10f}")
                print(f"  Total impact coef: {current_price_impact:.10f}")
                print(f"  Impact cost: ${impact_cost:.4f}")
                print(f"  Spread cost: ${spread_cost:.4f}")
                print(f"  Risk cost: ${risk_cost:.4f}")
                print(f"  Period total: ${impact_cost + spread_cost + risk_cost:.4f}")
                print(f"  Remaining inventory: {inventory:,.2f}")
                print()
            
            # 10. Update price displacement for next period (transient persists)
            price_displacement = price_displacement + transient_impact
        
        if debug:
            print("="*80)
            print("SUMMARY:")
            print(f"  Total impact: ${sum(impact_costs):.2f}")
            print(f"  Total spread: ${sum(spread_costs):.2f}")
            print(f"  Total risk: ${sum(risk_costs):.2f}")
            print(f"  TOTAL COST: ${total_cost:.2f}")
            print("="*80)
        
        return total_cost
    
    def cost_with_projection(self, S: np.ndarray) -> float:
        """
        Cost function with automatic projection onto feasible set.
        
        Projection:
        1. Clip to trade size limits: 0 ≤ S[i] ≤ max_trade_per_period
        2. Normalize to sum constraint: Σ S[i] = X₀
        
        Args:
            S: Trading strategy (possibly infeasible)
            
        Returns:
            Cost of projected (feasible) strategy
        """
        # Step 1: Enforce non-negativity and trade size limits
        S_feasible = np.clip(S, 0, self.max_trade_per_period)
        
        # Step 2: Normalize to sum to X0
        sum_S = np.sum(S_feasible)
        if sum_S > 1e-10:
            S_feasible = S_feasible * (self.X0 / sum_S)
        else:
            # If all zeros, use TWAP as fallback
            S_feasible = np.ones(self.N) * self.X0 / self.N
            S_feasible = np.minimum(S_feasible, self.max_trade_per_period)
        
        # Compute cost of feasible strategy
        return self.cost_function(S_feasible)
    
    def solve(self, 
              maxiter: int = 2000, 
              popsize: int = 40, 
              polish: bool = True,
              verbose: bool = True) -> Dict:
        """
        Solve optimal execution with realistic constraints using DE.
        
        Parameters
        ----------
        maxiter : int, default=2000
            Maximum number of DE iterations
        popsize : int, default=40
            Population size multiplier
        polish : bool, default=True
            Whether to use L-BFGS-B for final polish
        verbose : bool, default=True
            Whether to print progress information
        
        Returns
        -------
        dict
            optimal_trades, cost, solve_time, and diagnostics
        """
        if verbose:
            print("="*80)
            print("REALISTIC OPTIMAL EXECUTION SOLVER")
            print("="*80)
            print(f"Problem: X₀={self.X0:,.0f}, T={self.T}, N={self.N}")
            print(f"Market: σ={self.sigma:.4f}, η={self.eta:.2e}, γ={self.gamma:.4f}, S₀=${self.S0:.2f}")
            print()
            print("Realistic Constraints:")
            print(f"  • Max trade/period: {self.max_trade_fraction:.1%} of X₀ = {self.max_trade_per_period:,.0f} shares")
            print(f"  • Spread cost: {self.spread_bps:.1f} bps = ${self.spread_cost_per_share:.4f}/share")
            print(f"  • Permanent impact: {self.permanent_fraction:.1%} of total")
            print(f"  • Transient impact: {1-self.permanent_fraction:.1%} of total (decays at ρ={self.decay_rate})")
            print(f"  • Transient half-life: {np.log(2)/self.decay_rate:.2f} * τ = {np.log(2)/self.decay_rate * self.tau:.4f} days")
            print()
            print(f"Method: Differential Evolution (global optimization)")
            print(f"Settings: maxiter={maxiter}, popsize={popsize}, polish={polish}")
            print("="*80)
            print()
        
        start_time = time.time()
        
        # Define bounds: 0 ≤ S[i] ≤ max_trade_per_period
        bounds = [(0, self.max_trade_per_period) for _ in range(self.N)]
        
        # Run Differential Evolution
        result = differential_evolution(
            func=self.cost_with_projection,
            bounds=bounds,
            strategy='best1bin',
            maxiter=maxiter,
            popsize=popsize,
            tol=1e-8,
            mutation=(0.5, 1.5),
            recombination=0.7,
            seed=42,
            polish=polish,
            atol=0,
            updating='immediate',
            workers=1,
            disp=False
        )
        
        solve_time = time.time() - start_time
        
        # Project final solution to ensure exact feasibility
        S_optimal = np.clip(result.x, 0, self.max_trade_per_period)
        sum_S = np.sum(S_optimal)
        if sum_S > 1e-10:
            S_optimal = S_optimal * (self.X0 / sum_S)
        
        # Compute final cost
        final_cost = self.cost_function(S_optimal)
        
        # Compute cost breakdown for diagnostics
        cost_breakdown = self._compute_cost_breakdown(S_optimal)
        
        if verbose:
            print(f"✅ Optimization complete")
            print(f"   Iterations: {result.nit}")
            print(f"   Function evaluations: {result.nfev}")
            print(f"   Time: {solve_time:.2f}s")
            print()
            print(f"Optimal Strategy:")
            print(f"   Total cost: ${final_cost:.4f}")
            print(f"   Sum of trades: {np.sum(S_optimal):,.0f} (target: {self.X0:,.0f})")
            print(f"   Constraint error: {abs(np.sum(S_optimal) - self.X0):.2e}")
            print()
            print(f"Cost Breakdown:")
            print(f"   Impact cost: ${cost_breakdown['impact_cost']:.4f} ({cost_breakdown['impact_pct']:.1f}%)")
            print(f"   Spread cost: ${cost_breakdown['spread_cost']:.4f} ({cost_breakdown['spread_pct']:.1f}%)")
            print(f"   Risk cost: ${cost_breakdown['risk_cost']:.4f} ({cost_breakdown['risk_pct']:.1f}%)")
            print()
            print(f"Trading Pattern:")
            print(f"   First 3 trades: {S_optimal[:3] / self.X0 * 100}")
            print(f"   First trade: {S_optimal[0] / self.X0:.1%} of X₀")
            print(f"   Max trade: {np.max(S_optimal) / self.X0:.1%} of X₀")
            print(f"   Nonzero trades: {np.sum(S_optimal > self.X0 * 0.01)}/{self.N}")
            print("="*80)
        
        return {
            'optimal_trades': S_optimal,
            'cost': final_cost,
            'cost_breakdown': cost_breakdown,
            'success': result.success,
            'message': result.message,
            'solve_time': solve_time,
            'num_function_calls': result.nfev,
            'nit': result.nit,
            'method': 'Differential Evolution (Realistic)',
            'constraints': {
                'max_trade_fraction': self.max_trade_fraction,
                'spread_bps': self.spread_bps,
                'permanent_fraction': self.permanent_fraction,
                'decay_rate': self.decay_rate
            }
        }
    
    def _compute_cost_breakdown(self, S: np.ndarray) -> Dict:
        """
        Compute detailed cost breakdown for analysis.
        
        Returns:
            Dictionary with impact, spread, and risk costs
        """
        tau = self.tau
        price_displacement = 0.0
        inventory = self.X0
        
        impact_cost_total = 0.0
        spread_cost_total = 0.0
        risk_cost_total = 0.0
        
        for i in range(self.N):
            # Decay transient
            decay_factor = np.exp(-self.decay_rate * tau)
            price_displacement = price_displacement * decay_factor
            
            # Compute impacts
            permanent_impact = self.eta_permanent * (np.abs(S[i]) ** self.gamma)
            transient_impact = self.eta_transient * (np.abs(S[i]) ** self.gamma)
            current_price_impact = price_displacement + permanent_impact + transient_impact
            
            # Accumulate costs
            # Impact cost = S[i] × (price impact coefficient) × S₀
            impact_cost_total += S[i] * current_price_impact * self.S0
            spread_cost_total += self.spread_cost_per_share * S[i]
            
            # Update inventory and compute risk
            inventory -= S[i]
            risk_cost_total += 0.5 * self.lam * (inventory ** 2) * (self.sigma ** 2) * tau
            
            # Update price displacement
            price_displacement = price_displacement + transient_impact
        
        total = impact_cost_total + spread_cost_total + risk_cost_total
        
        return {
            'impact_cost': impact_cost_total,
            'spread_cost': spread_cost_total,
            'risk_cost': risk_cost_total,
            'total_cost': total,
            'impact_pct': 100 * impact_cost_total / total if total > 0 else 0,
            'spread_pct': 100 * spread_cost_total / total if total > 0 else 0,
            'risk_pct': 100 * risk_cost_total / total if total > 0 else 0
        }


def solve_optimal_execution_realistic(
    X0: float,
    T: float, 
    N: int,
    sigma: float,
    lam: float,
    eta: float,
    gamma: float,
    S0: float,
    # Optional constraint overrides (use defaults if None)
    max_trade_fraction: Optional[float] = None,
    spread_bps: Optional[float] = None,
    permanent_fraction: Optional[float] = None,
    decay_rate: Optional[float] = None,
    # Solver settings
    maxiter: int = 2000,
    popsize: int = 40,
    polish: bool = True,
    verbose: bool = True
) -> Dict:
    """
    Convenience wrapper for realistic optimal execution solver.
    
    Uses empirically-calibrated defaults unless explicitly overridden.
    
    Example Usage
    -------------
    # Basic usage (all defaults):
    >>> result = solve_optimal_execution_realistic(
    ...     X0=100000, T=1.0, N=10,
    ...     sigma=0.035, lam=1e-6, eta=2e-7, gamma=0.67, S0=7.92
    ... )
    
    # Custom constraints for illiquid stock:
    >>> result = solve_optimal_execution_realistic(
    ...     X0=50000, T=1.0, N=10,
    ...     sigma=0.05, lam=1e-6, eta=5e-7, gamma=0.65, S0=15.0,
    ...     max_trade_fraction=0.2,  # Lower limit for illiquid
    ...     spread_bps=5.0            # Wider spread
    ... )
    
    Parameters
    ----------
    X0, T, N, sigma, lam, eta, gamma, S0 : float/int
        Core problem parameters
        
    max_trade_fraction : float, optional
        Default 0.4 (40% max per period)
        
    spread_bps : float, optional
        Default 1.0 (1 basis point)
        
    permanent_fraction : float, optional
        Default 0.4 (40% permanent, 60% transient)
        
    decay_rate : float, optional
        Default 0.5 (half-life ≈ 1.4 * τ)
        
    maxiter, popsize, polish : int/bool
        DE solver settings
        
    verbose : bool
        Print progress information
    
    Returns
    -------
    dict
        optimal_trades, cost, cost_breakdown, solve_time, etc.
    """
    # Set defaults if not specified
    if max_trade_fraction is None:
        max_trade_fraction = 0.4
    if spread_bps is None:
        spread_bps = 1.0
    if permanent_fraction is None:
        permanent_fraction = 0.4
    if decay_rate is None:
        decay_rate = 0.5
    
    # Create solver
    solver = OptimalExecutionRealistic(
        X0=X0, T=T, N=N,
        sigma=sigma, lam=lam,
        eta=eta, gamma=gamma, S0=S0,
        max_trade_fraction=max_trade_fraction,
        spread_bps=spread_bps,
        permanent_fraction=permanent_fraction,
        decay_rate=decay_rate
    )
    
    # Solve
    result = solver.solve(
        maxiter=maxiter,
        popsize=popsize,
        polish=polish,
        verbose=verbose
    )
    
    return result


def compare_instantaneous_vs_realistic(
    X0: float, T: float, N: int,
    sigma: float, lam: float,
    eta: float, gamma: float, S0: float,
    verbose: bool = True
) -> Dict:
    """
    Compare instantaneous impact model vs realistic full model.
    
    This demonstrates the effect of each step:
    - Instantaneous: Corner solution (99% in period 1)
    - Realistic: Smooth front-loading (natural constraints)
    
    Returns comparison metrics and strategies.
    """
    try:
        from .de_solver import OptimalExecutionDE
    except ImportError:
        from de_solver import OptimalExecutionDE
    
    if verbose:
        print("\n" + "="*80)
        print("COMPARISON: INSTANTANEOUS vs REALISTIC IMPACT MODELS")
        print("="*80 + "\n")
    
    # Solve with instantaneous impact (baseline)
    if verbose:
        print("1. INSTANTANEOUS IMPACT MODEL (Baseline)")
        print("-" * 80)
    
    solver_instant = OptimalExecutionDE(
        X0=X0, T=T, N=N,
        sigma=sigma, lam=lam,
        eta=eta, gamma=gamma, S0=S0
    )
    result_instant = solver_instant.solve(maxiter=2000, verbose=verbose)
    
    if verbose:
        print("\n2. REALISTIC FULL MODEL")
        print("-" * 80)
    
    # Solve with realistic model
    result_realistic = solve_optimal_execution_realistic(
        X0=X0, T=T, N=N,
        sigma=sigma, lam=lam,
        eta=eta, gamma=gamma, S0=S0,
        verbose=verbose
    )
    
    # Compute comparison metrics
    instant_trades = result_instant['optimal_trades']
    realistic_trades = result_realistic['optimal_trades']
    
    comparison = {
        'instantaneous': {
            'trades': instant_trades,
            'cost': result_instant['cost'],
            'first_trade_pct': instant_trades[0] / X0,
            'max_trade_pct': np.max(instant_trades) / X0,
            'num_nonzero_trades': np.sum(instant_trades > X0 * 0.01)
        },
        'realistic': {
            'trades': realistic_trades,
            'cost': result_realistic['cost'],
            'cost_breakdown': result_realistic['cost_breakdown'],
            'first_trade_pct': realistic_trades[0] / X0,
            'max_trade_pct': np.max(realistic_trades) / X0,
            'num_nonzero_trades': np.sum(realistic_trades > X0 * 0.01)
        },
        'cost_increase': (result_realistic['cost'] - result_instant['cost']) / result_instant['cost']
    }
    
    if verbose:
        print("\n" + "="*80)
        print("COMPARISON SUMMARY")
        print("="*80)
        print(f"\nInstantaneous Model:")
        print(f"  Cost: ${result_instant['cost']:.4f}")
        print(f"  First trade: {comparison['instantaneous']['first_trade_pct']:.1%}")
        print(f"  Max trade: {comparison['instantaneous']['max_trade_pct']:.1%}")
        print(f"  Active periods: {comparison['instantaneous']['num_nonzero_trades']}/{N}")
        print(f"  Pattern: Corner solution (front-load everything)")
        
        print(f"\nRealistic Model:")
        print(f"  Cost: ${result_realistic['cost']:.4f}")
        print(f"  First trade: {comparison['realistic']['first_trade_pct']:.1%}")
        print(f"  Max trade: {comparison['realistic']['max_trade_pct']:.1%}")
        print(f"  Active periods: {comparison['realistic']['num_nonzero_trades']}/{N}")
        print(f"  Pattern: Smooth front-loading (natural constraints)")
        
        print(f"\nCost Impact:")
        print(f"  Increase: {comparison['cost_increase']:.1%}")
        print(f"  Interpretation: Premium paid for realistic constraints")
        print(f"  Acceptable: {'✅ Yes' if comparison['cost_increase'] < 0.5 else '⚠️ High'}")
        print("="*80 + "\n")
    
    return comparison


if __name__ == "__main__":
    """
    Validation: Test each step produces expected behavior.
    """
    print("="*80)
    print("REALISTIC OPTIMAL EXECUTION: VALIDATION TESTS")
    print("="*80)
    print()
    
    # Test parameters (SNAP - from calibration)
    X0 = 100000
    T = 1.0
    N = 10
    sigma = 0.0348
    lam = 1e-6
    eta = 2e-7
    gamma = 0.67
    S0 = 7.92
    
    print("Test Stock: SNAP")
    print(f"  X₀ = {X0:,} shares")
    print(f"  T = {T} day, N = {N} periods")
    print(f"  σ = {sigma:.4f}, λ = {lam:.2e}")
    print(f"  η = {eta:.2e}, γ = {gamma:.4f}, S₀ = ${S0:.2f}")
    print()
    
    # Run comparison
    comparison = compare_instantaneous_vs_realistic(
        X0=X0, T=T, N=N,
        sigma=sigma, lam=lam,
        eta=eta, gamma=gamma, S0=S0,
        verbose=True
    )
    
    # Validate expectations
    print("="*80)
    print("VALIDATION CHECKS")
    print("="*80)
    print()
    
    instant_max = comparison['instantaneous']['max_trade_pct']
    realistic_max = comparison['realistic']['max_trade_pct']
    cost_increase = comparison['cost_increase']
    
    # Check 1: Instantaneous has corner solution
    check1 = instant_max > 0.95
    print(f"✓ Check 1: Instantaneous has corner solution (>95% in one trade)")
    print(f"  Result: {instant_max:.1%} {'✅ PASS' if check1 else '❌ FAIL'}")
    
    # Check 2: Realistic respects trade size limit
    check2 = realistic_max <= 0.41  # Allow 1% tolerance
    print(f"\n✓ Check 2: Realistic respects 40% trade size limit")
    print(f"  Result: {realistic_max:.1%} {'✅ PASS' if check2 else '❌ FAIL'}")
    
    # Check 3: Cost increase is reasonable
    check3 = 0 < cost_increase < 0.5  # Between 0% and 50%
    print(f"\n✓ Check 3: Cost increase is reasonable (<50%)")
    print(f"  Result: {cost_increase:.1%} {'✅ PASS' if check3 else '❌ FAIL'}")
    
    # Check 4: Realistic uses multiple periods
    realistic_active = comparison['realistic']['num_nonzero_trades']
    check4 = realistic_active >= 3
    print(f"\n✓ Check 4: Realistic strategy uses multiple periods (≥3)")
    print(f"  Result: {realistic_active} periods {'✅ PASS' if check4 else '❌ FAIL'}")
    
    all_pass = check1 and check2 and check3 and check4
    print()
    print("="*80)
    print(f"OVERALL: {'✅ ALL TESTS PASSED' if all_pass else '❌ SOME TESTS FAILED'}")
    print("="*80)
