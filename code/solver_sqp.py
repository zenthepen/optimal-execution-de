"""
SQP-Based Optimal Execution Solver

Implementation of Sequential Quadratic Programming for optimal execution
with instantaneous power-law market impact. This provides an alternative
to the DP solver with potentially better handling of continuous optimization.

Key Features:
- Instantaneous impact model: η × |S|^γ × S₀
- No transient/permanent decomposition (same as DP solver)
- Multiple initial guesses for robustness
- SLSQP optimization with explicit constraints

Author: Generated for optimal execution project
Date: 2025-10-28
"""

import numpy as np
import time
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Optional


class OptimalExecutionSQP:
    """
    SQP-based solver for optimal execution with power-law impact.
    
    Mathematical Formulation:
    -------------------------
    minimize: Σᵢ [η|Sᵢ|^γ·S₀ + 0.5·λ·σ²·xᵢ²·τ]
    
    subject to:
        Σᵢ Sᵢ = X₀  (complete liquidation)
        Sᵢ ≥ 0      (no buying)
        
    where:
        xᵢ = X₀ - Σⱼ₌₁ⁱ Sⱼ  (inventory after trade i)
        τ = T/N              (time step)
    """
    
    def __init__(self, X0: float, T: float, N: int, sigma: float, lam: float, 
                 eta: float, gamma: float, S0: float):
        """
        Initialize optimal execution problem.
        
        Args:
            X0: Initial inventory (shares)
            T: Time horizon (days)
            N: Number of trading periods
            sigma: Volatility (daily)
            lam: Risk aversion parameter
            eta: Impact coefficient
            gamma: Impact power-law exponent
            S0: Stock price (for dollar impact calculation)
        """
        self.X0 = X0
        self.T = T
        self.N = N
        self.sigma = sigma
        self.lam = lam
        self.eta = eta
        self.gamma = gamma
        self.S0 = S0
        self.tau = T / N  # Time step size
        
    def cost_function(self, S: np.ndarray) -> float:
        """
        Compute total expected cost for trading strategy S.
        
        CRITICAL: Uses inventory AFTER trade for volatility cost.
        This matches the fixed DP solver implementation.
        
        Args:
            S: Trading strategy [S₁, S₂, ..., Sₙ]
            
        Returns:
            Total expected cost (impact + risk)
        """
        total_cost = 0.0
        inventory = self.X0
        
        for i in range(self.N):
            # Impact cost (instantaneous, paid at execution)
            impact = self.eta * (np.abs(S[i]) ** self.gamma) * self.S0
            
            # Update inventory AFTER trade
            inventory -= S[i]
            
            # Risk cost (inventory held during period)
            # Uses inventory AFTER trade (correct per Almgren-Chriss)
            risk = 0.5 * self.lam * (inventory ** 2) * (self.sigma ** 2) * self.tau
            
            total_cost += impact + risk
            
        return total_cost
    
    def cost_gradient(self, S: np.ndarray) -> np.ndarray:
        """
        Compute gradient of cost function for faster optimization.
        
        ∂C/∂Sᵢ = η·γ·|Sᵢ|^(γ-1)·S₀·sign(Sᵢ) - λ·σ²·τ·Σⱼ≥ᵢ xⱼ
        
        Args:
            S: Trading strategy
            
        Returns:
            Gradient vector ∂C/∂S
        """
        grad = np.zeros(self.N)
        
        # Compute inventories after each trade
        inventories = self.X0 - np.cumsum(S)
        
        for i in range(self.N):
            # Impact gradient
            if abs(S[i]) > 1e-12:
                impact_grad = self.eta * self.gamma * (np.abs(S[i]) ** (self.gamma - 1)) * \
                             self.S0 * np.sign(S[i])
            else:
                # Handle S[i] ≈ 0 to avoid numerical issues
                impact_grad = 0.0
            
            # Risk gradient (affects all future periods)
            # ∂/∂Sᵢ [Σⱼ≥ᵢ 0.5·λ·σ²·xⱼ²·τ] = -λ·σ²·τ·Σⱼ≥ᵢ xⱼ
            risk_grad = -self.lam * (self.sigma ** 2) * self.tau * np.sum(inventories[i:])
            
            grad[i] = impact_grad + risk_grad
            
        return grad
    
    def constraint_complete_liquidation(self, S: np.ndarray) -> float:
        """
        Equality constraint: sum(S) - X₀ = 0
        
        Args:
            S: Trading strategy
            
        Returns:
            Constraint violation (should be 0)
        """
        return np.sum(S) - self.X0
    
    def constraint_gradient(self, S: np.ndarray) -> np.ndarray:
        """Gradient of liquidation constraint: all ones."""
        return np.ones(self.N)
    
    def generate_initial_guesses(self, num_guesses: int = 5) -> List[np.ndarray]:
        """
        Generate diverse initial guesses for multi-start optimization.
        
        Args:
            num_guesses: Number of initial guesses to generate (default 5, can use 50+ for robustness)
        
        Returns:
            List of initial strategies, each summing to X₀
        """
        guesses = []
        
        # 1. TWAP (uniform)
        twap = np.ones(self.N) * self.X0 / self.N
        guesses.append(twap)
        
        # 2. Front-loaded (exponential decay)
        decay_factor = 0.9
        front = np.array([self.X0 * (1 - decay_factor) * (decay_factor ** i) 
                         for i in range(self.N)])
        front = front * self.X0 / (np.sum(front) + 1e-10)
        guesses.append(front)
        
        # 3. Back-loaded (reverse of front-loaded)
        back = front[::-1].copy()
        guesses.append(back)
        
        # 4. Middle-peaked (Gaussian)
        center = self.N / 2
        width = self.N / 4
        middle = np.array([np.exp(-0.5 * ((i - center) / width) ** 2) 
                          for i in range(self.N)])
        middle = middle * self.X0 / (np.sum(middle) + 1e-10)
        guesses.append(middle)
        
        # 5. Random monotone (exponential distribution)
        rng = np.random.default_rng(42)
        random = rng.exponential(self.X0 / self.N, self.N)
        random = random * self.X0 / (np.sum(random) + 1e-10)
        guesses.append(random)
        
        # If more guesses requested, add random variations
        if num_guesses > 5:
            rng = np.random.default_rng(42)
            
            for i in range(num_guesses - 5):
                # Generate different types of random guesses
                guess_type = i % 5
                
                if guess_type == 0:
                    # Random exponential with varying rate
                    rate = rng.uniform(0.5, 2.0)
                    guess = rng.exponential(self.X0 / (self.N * rate), self.N)
                elif guess_type == 1:
                    # Random Gaussian mixture
                    center1 = rng.uniform(0, self.N)
                    width1 = rng.uniform(self.N / 6, self.N / 3)
                    guess = np.array([np.exp(-0.5 * ((j - center1) / width1) ** 2) 
                                     for j in range(self.N)])
                elif guess_type == 2:
                    # Random Dirichlet (ensures sum = 1, then scale)
                    alpha = rng.uniform(0.1, 2.0, self.N)
                    guess = rng.dirichlet(alpha)  * self.X0
                elif guess_type == 3:
                    # Random power-law
                    alpha = rng.uniform(0.5, 2.0)
                    guess = np.array([(j + 1) ** (-alpha) for j in range(self.N)])
                else:
                    # Random monotone decreasing
                    guess = np.sort(rng.uniform(0, 1, self.N))[::-1]
                
                # Normalize to sum to X0
                guess = guess * self.X0 / (np.sum(guess) + 1e-10)
                guesses.append(guess)
        
        return guesses
    
    def solve_from_guess(self, S0_guess: np.ndarray, verbose: bool = False) -> 'minimize':
        """
        Solve optimization from single initial guess using SLSQP.
        
        Args:
            S0_guess: Initial guess for trades
            verbose: Print optimization progress
            
        Returns:
            scipy.optimize.OptimizeResult object
        """
        result = minimize(
            fun=self.cost_function,
            x0=S0_guess,
            method='SLSQP',
            jac=self.cost_gradient,  # Provide gradient for faster convergence
            bounds=[(0, self.X0) for _ in range(self.N)],
            constraints=[{
                'type': 'eq',
                'fun': self.constraint_complete_liquidation,
                'jac': self.constraint_gradient
            }],
            options={
                'maxiter': 1000,
                'ftol': 1e-9,
                'disp': verbose
            }
        )
        
        return result
    
    def solve(self, num_guesses: int = 5, verbose: bool = True) -> Dict:
        """
        Solve using multiple starts. Try all initial guesses, return best result.
        
        Args:
            num_guesses: Number of initial guesses to try (default 5, use 50+ for robustness)
            verbose: Print progress information
            
        Returns:
            Dictionary with solution details
        """
        if verbose:
            print("="*70)
            print("SQP OPTIMAL EXECUTION SOLVER")
            print("="*70)
            print(f"Problem: X₀={self.X0:,.0f}, T={self.T}, N={self.N}")
            print(f"Market: σ={self.sigma:.4f}, η={self.eta:.2e}, γ={self.gamma:.4f}")
            print(f"Strategy: Multi-start optimization with {num_guesses} initial guesses")
            print()
        
        start_time = time.time()
        initial_guesses = self.generate_initial_guesses(num_guesses=num_guesses)
        
        best_result = None
        best_cost = np.inf
        best_guess_idx = -1
        
        # Try all initial guesses
        for idx, guess in enumerate(initial_guesses):
            if verbose and (idx < 5 or idx % 10 == 0 or idx == len(initial_guesses) - 1):
                if idx < 5:
                    guess_names = ["TWAP", "Front-loaded", "Back-loaded", 
                                  "Middle-peaked", "Random"]
                    print(f"Trying guess #{idx+1}/{num_guesses}: {guess_names[idx]}...")
                else:
                    print(f"Trying guess #{idx+1}/{num_guesses}...")
            
            result = self.solve_from_guess(guess, verbose=False)
            
            if verbose and (idx < 5 or idx % 10 == 0 or idx == len(initial_guesses) - 1):
                if result.success:
                    print(f"  ✅ Success: cost = ${result.fun:.2f}")
                else:
                    print(f"  ❌ Failed: {result.message}")
            
            # Update best result
            if result.success and result.fun < best_cost:
                best_cost = result.fun
                best_result = result
                best_guess_idx = idx
        
        solve_time = time.time() - start_time
        
        if verbose:
            print()
            if best_result is not None and best_result.success:
                print(f"✅ Best solution found from guess #{best_guess_idx+1}/{num_guesses}")
                print(f"   Cost: ${best_cost:.2f}")
                print(f"   Time: {solve_time:.2f}s")
            else:
                print("❌ No successful solution found")
            print("="*70)
        
        # Prepare return dictionary
        if best_result is not None:
            return {
                'optimal_trades': best_result.x,
                'cost': best_result.fun,
                'success': best_result.success,
                'message': best_result.message,
                'solve_time': solve_time,
                'num_function_calls': best_result.nfev,
                'initial_guess_index': best_guess_idx,
                'num_guesses_tried': num_guesses,
                'nit': best_result.nit,
                'method': f'SLSQP (multi-start {num_guesses})'
            }
        else:
            # All guesses failed
            return {
                'optimal_trades': None,
                'cost': np.inf,
                'success': False,
                'message': 'All initial guesses failed',
                'solve_time': solve_time,
                'num_function_calls': 0,
                'initial_guess_index': -1,
                'num_guesses_tried': num_guesses,
                'nit': 0,
                'method': f'SLSQP (multi-start {num_guesses})'
            }
    
    def validate_solution(self, trades: np.ndarray) -> Dict:
        """
        Validate solution satisfies constraints and bounds.
        
        Args:
            trades: Proposed trading strategy
            
        Returns:
            Dictionary with validation results
        """
        sum_trades = np.sum(trades)
        sum_error = abs(sum_trades - self.X0)
        min_trade = np.min(trades)
        max_trade = np.max(trades)
        
        all_nonnegative = bool(np.all(trades >= -1e-8))
        sum_constraint_satisfied = bool(np.isclose(sum_trades, self.X0, atol=1e-6))
        is_valid = all_nonnegative and sum_constraint_satisfied
        
        return {
            'sum_trades': sum_trades,
            'sum_error': sum_error,
            'min_trade': min_trade,
            'max_trade': max_trade,
            'all_nonnegative': all_nonnegative,
            'sum_constraint_satisfied': sum_constraint_satisfied,
            'is_valid': is_valid
        }
    
    def compare_to_twap(self, optimal_trades: np.ndarray) -> Dict:
        """
        Compare optimal solution to TWAP benchmark.
        
        Args:
            optimal_trades: Optimal trading strategy
            
        Returns:
            Comparison metrics
        """
        twap_trades = np.ones(self.N) * self.X0 / self.N
        
        optimal_cost = self.cost_function(optimal_trades)
        twap_cost = self.cost_function(twap_trades)
        
        improvement_dollars = twap_cost - optimal_cost
        improvement_pct = (improvement_dollars / twap_cost) * 100 if twap_cost > 0 else 0
        
        return {
            'optimal_cost': optimal_cost,
            'twap_cost': twap_cost,
            'improvement_dollars': improvement_dollars,
            'improvement_percent': improvement_pct,
            'twap_trades': twap_trades
        }


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def solve_optimal_execution_sqp(X0: float, T: float, N: int, sigma: float, 
                               lam: float, eta: float, gamma: float, S0: float,
                               num_guesses: int = 5, verbose: bool = True) -> Dict:
    """
    Convenience wrapper for SQP solver matching DP interface.
    
    Args:
        X0: Initial inventory (shares)
        T: Time horizon (days)
        N: Number of trading periods
        sigma: Volatility
        lam: Risk aversion
        eta: Impact coefficient
        gamma: Impact exponent
        S0: Stock price
        num_guesses: Number of initial guesses for multi-start (default 5, use 50 for robustness)
        verbose: Print progress
        
    Returns:
        Solution dictionary
    """
    solver = OptimalExecutionSQP(X0, T, N, sigma, lam, eta, gamma, S0)
    return solver.solve(num_guesses=num_guesses, verbose=verbose)


def compute_twap_cost(X0: float, T: float, N: int, sigma: float, lam: float,
                     eta: float, gamma: float, S0: float) -> float:
    """
    Compute cost of TWAP strategy for comparison.
    
    Args:
        (same as solve_optimal_execution_sqp)
        
    Returns:
        TWAP strategy cost
    """
    twap_trades = np.ones(N) * X0 / N
    solver = OptimalExecutionSQP(X0, T, N, sigma, lam, eta, gamma, S0)
    return solver.cost_function(twap_trades)


# ============================================================================
# TESTING AND VALIDATION
# ============================================================================

def run_snap_example():
    """Run example on SNAP parameters (matching your calibration)."""
    print("\n" + "="*70)
    print("EXAMPLE: SNAP OPTIMAL EXECUTION (SQP)")
    print("="*70 + "\n")
    
    # SNAP parameters from your calibration
    result = solve_optimal_execution_sqp(
        X0=100000,      # 100k shares
        T=1.0,          # 1 day
        N=10,           # 10 periods
        sigma=0.0348,   # SNAP volatility
        lam=1e-6,       # Risk aversion
        eta=2e-7,       # Impact coefficient (calibrated)
        gamma=0.67,     # Power-law exponent (calibrated)
        S0=7.92,        # SNAP price
        verbose=True
    )
    
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"Success: {result['success']}")
    print(f"Optimal cost: ${result['cost']:.2f}")
    print(f"Solve time: {result['solve_time']:.3f}s")
    print(f"Function calls: {result['num_function_calls']}")
    print(f"Iterations: {result['nit']}")
    print(f"Best initial guess: #{result['initial_guess_index']+1}")
    print()
    
    if result['success']:
        print("Optimal Trading Strategy:")
        for i, trade in enumerate(result['optimal_trades']):
            print(f"  Period {i+1}: {trade:,.0f} shares ({trade/1000:.1f}k)")
        print(f"  Total: {np.sum(result['optimal_trades']):,.0f} shares")
        print()
        
        # Validate solution
        solver = OptimalExecutionSQP(100000, 1.0, 10, 0.0348, 1e-6, 2e-7, 0.67, 7.92)
        validation = solver.validate_solution(result['optimal_trades'])
        print("Validation:")
        print(f"  Sum constraint error: {validation['sum_error']:.6f}")
        print(f"  All non-negative: {validation['all_nonnegative']}")
        print(f"  Solution valid: {validation['is_valid']} ✅")
        print()
        
        # Compare to TWAP
        comparison = solver.compare_to_twap(result['optimal_trades'])
        print("Comparison to TWAP:")
        print(f"  TWAP cost: ${comparison['twap_cost']:.2f}")
        print(f"  Optimal cost: ${comparison['optimal_cost']:.2f}")
        print(f"  Improvement: ${comparison['improvement_dollars']:.2f} "
              f"({comparison['improvement_percent']:.2f}%)")
    
    print("="*70 + "\n")
    
    return result


if __name__ == "__main__":
    # Run SNAP example
    result = run_snap_example()
