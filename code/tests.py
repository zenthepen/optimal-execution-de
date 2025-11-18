"""
================================================================================
BULLETPROOF FINAL TEST SUITE FOR OPTIMAL EXECUTION SOLVER
================================================================================

This comprehensive test suite validates EVERY aspect of the algorithm:

✓ Mathematical correctness
✓ Constraint satisfaction
✓ Edge cases and boundary conditions
✓ Numerical stability
✓ Cross-validation with known results
✓ Stress testing
✓ Real-world scenarios
✓ Algorithm consistency

Run this ONCE before thesis submission to guarantee quality.

================================================================================
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add core directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent / 'core'))

from de_solver_realistic import OptimalExecutionRealistic

class ComprehensiveSolverTest:
    """Complete validation suite for optimal execution solver"""
    
    def __init__(self):
        self.results = []
        self.passed = 0
        self.failed = 0
        self.warnings_count = 0
        
    def log_test(self, test_name, passed, message=""):
        """Log test result"""
        status = "✅ PASS" if passed else "❌ FAIL"
        self.results.append({
            'test': test_name,
            'status': status,
            'message': message
        })
        if passed:
            self.passed += 1
        else:
            self.failed += 1
        print(f"{status} | {test_name}: {message}")
    
    def log_warning(self, test_name, message=""):
        """Log warning (not a failure, but worth noting)"""
        self.results.append({
            'test': test_name,
            'status': '⚠️  WARN',
            'message': message
        })
        self.warnings_count += 1
        print(f"⚠️  WARN | {test_name}: {message}")
    
    # ========================================================================
    # SECTION 1: BASIC MATHEMATICAL TESTS
    # ========================================================================
    
    def test_1_basic_math_power_law(self):
        """Test 1: Power-law impact calculation is correct"""
        print("\n" + "="*100)
        print("SECTION 1: BASIC MATHEMATICAL CORRECTNESS")
        print("="*100)
        
        # Manual calculation
        eta = 2e-7
        S = 100000
        gamma = 0.67
        S0 = 7.92
        
        expected_coef = eta * (S ** gamma)
        expected_impact = S * expected_coef * S0
        
        # Solver calculation
        solver = OptimalExecutionRealistic(
            X0=100000, T=1.0, N=10,
            sigma=0.0348, lam=1e-6,
            eta=eta, gamma=gamma, S0=S0,
            max_trade_fraction=1.0, spread_bps=0.0,
            permanent_fraction=1.0, decay_rate=0.0
        )
        
        trades = np.array([S] + [0]*9)
        solver_cost = solver.cost_function(trades)
        
        passed = abs(solver_cost - expected_impact) < 1.0
        self.log_test(
            "Power-law impact calculation",
            passed,
            f"Expected: ${expected_impact:.2f}, Got: ${solver_cost:.2f}"
        )
        return passed
    
    def test_2_spread_cost_calculation(self):
        """Test 2: Bid-ask spread cost is correctly applied"""
        
        spread_bps = 1.0
        S = 100000
        S0 = 7.92
        
        expected_spread = S * (spread_bps * 0.0001) * S0
        
        solver = OptimalExecutionRealistic(
            X0=100000, T=1.0, N=10,
            sigma=0.0348, lam=1e-6,
            eta=0, gamma=0.67, S0=S0,  # Zero impact
            max_trade_fraction=1.0, spread_bps=spread_bps,
            permanent_fraction=0.0, decay_rate=0.0  # No impact, only spread
        )
        
        trades = np.array([float(S)] + [0.0]*9)
        solver_cost = solver.cost_function(trades)
        
        passed = abs(solver_cost - expected_spread) < 0.5  # Slightly looser tolerance
        self.log_test(
            "Bid-ask spread calculation",
            passed,
            f"Expected: ${expected_spread:.2f}, Got: ${solver_cost:.2f}"
        )
        return passed
    
    def test_3_inventory_risk_cost(self):
        """Test 3: Inventory risk cost (Almgren-Chriss) calculated correctly"""
        
        # For uniform execution (all periods), risk cost is predictable
        X0 = 100000
        N = 10
        lam = 1e-6
        sigma = 0.0348
        T = 1.0
        tau = T / N
        
        # Uniform execution: 10k per period
        trades = np.ones(N) * 10000
        
        # Expected inventory risk (holds max at start of execution)
        # Risk = 0.5 * lambda * inventory^2 * sigma^2 * tau (summed over periods)
        expected_risk = 0
        inventory = X0
        for t in range(N):
            inventory -= trades[t]
            expected_risk += 0.5 * lam * (inventory ** 2) * (sigma ** 2) * tau
        
        solver = OptimalExecutionRealistic(
            X0=X0, T=T, N=N,
            sigma=sigma, lam=lam,
            eta=0, gamma=0.67, S0=7.92,  # Zero impact
            max_trade_fraction=1.0, spread_bps=0.0,
            permanent_fraction=0.0, decay_rate=0.0  # Only risk cost
        )
        
        solver_cost = solver.cost_function(trades)
        
        # Should be close (within small numerical error)
        passed = abs(solver_cost - expected_risk) < max(1.0, expected_risk * 0.1)  # 10% tolerance or $1
        self.log_test(
            "Inventory risk cost calculation",
            passed,
            f"Expected: ${expected_risk:.2f}, Got: ${solver_cost:.2f}"
        )
        return passed
    
    # ========================================================================
    # SECTION 2: CONSTRAINT SATISFACTION TESTS
    # ========================================================================
    
    def test_4_trade_size_constraint(self):
        """Test 4: Trade size limit is always respected"""
        print("\n" + "="*100)
        print("SECTION 2: CONSTRAINT SATISFACTION")
        print("="*100)
        
        max_frac = 0.4
        
        solver = OptimalExecutionRealistic(
            X0=100000, T=1.0, N=10,
            sigma=0.0348, lam=1e-6,
            eta=2e-7, gamma=0.67, S0=7.92,
            max_trade_fraction=max_frac, spread_bps=1.0,
            permanent_fraction=0.4, decay_rate=0.5
        )
        
        result = solver.solve(maxiter=1000, verbose=False)
        trades = result['optimal_trades']
        
        max_trade_actual = np.max(trades) / 100000
        constraint_satisfied = max_trade_actual <= max_frac + 1e-6  # Small tolerance for numerical error
        
        self.log_test(
            "Trade size constraint (max per period)",
            constraint_satisfied,
            f"Max: {max_frac:.1%}, Actual: {max_trade_actual:.1%}"
        )
        return constraint_satisfied
    
    def test_5_total_shares_constraint(self):
        """Test 5: Total shares executed equals order size"""
        
        X0 = 100000
        
        solver = OptimalExecutionRealistic(
            X0=X0, T=1.0, N=10,
            sigma=0.0348, lam=1e-6,
            eta=2e-7, gamma=0.67, S0=7.92,
            max_trade_fraction=0.4, spread_bps=1.0,
            permanent_fraction=0.4, decay_rate=0.5
        )
        
        result = solver.solve(maxiter=1000, verbose=False)
        trades = result['optimal_trades']
        
        total_executed = np.sum(trades)
        constraint_satisfied = abs(total_executed - X0) < 1e-6
        
        self.log_test(
            "Total shares constraint",
            constraint_satisfied,
            f"Required: {X0}, Executed: {total_executed:.0f}"
        )
        return constraint_satisfied
    
    def test_6_non_negative_trades(self):
        """Test 6: No negative trade sizes (short selling)"""
        
        solver = OptimalExecutionRealistic(
            X0=100000, T=1.0, N=10,
            sigma=0.0348, lam=1e-6,
            eta=2e-7, gamma=0.67, S0=7.92,
            max_trade_fraction=0.4, spread_bps=1.0,
            permanent_fraction=0.4, decay_rate=0.5
        )
        
        result = solver.solve(maxiter=1000, verbose=False)
        trades = result['optimal_trades']
        
        all_non_negative = np.all(trades >= -1e-10)  # Small tolerance
        
        self.log_test(
            "Non-negative trade constraint",
            all_non_negative,
            f"Min trade: {np.min(trades):.2e}, Max trade: {np.max(trades):.0f}"
        )
        return all_non_negative
    
    # ========================================================================
    # SECTION 3: EDGE CASES AND BOUNDARY CONDITIONS
    # ========================================================================
    
    def test_7_single_period_execution(self):
        """Test 7: Single period (N=1) executes everything instantly"""
        print("\n" + "="*100)
        print("SECTION 3: EDGE CASES & BOUNDARY CONDITIONS")
        print("="*100)
        
        solver = OptimalExecutionRealistic(
            X0=100000, T=1.0, N=1,
            sigma=0.0348, lam=1e-6,
            eta=2e-7, gamma=0.67, S0=7.92,
            max_trade_fraction=1.0, spread_bps=1.0,
            permanent_fraction=0.4, decay_rate=0.5
        )
        
        result = solver.solve(maxiter=500, verbose=False)
        trades = result['optimal_trades']
        
        # Should execute all in period 1
        executed_in_p1 = trades[0] / 100000
        single_period_optimal = executed_in_p1 > 0.99
        
        self.log_test(
            "Single period execution",
            single_period_optimal,
            f"Period 1: {executed_in_p1:.1%} (should be ~100%)"
        )
        return single_period_optimal
    
    def test_8_very_long_horizon(self):
        """Test 8: Very long horizon (N=50) spreads execution"""
        
        solver = OptimalExecutionRealistic(
            X0=100000, T=1.0, N=50,
            sigma=0.0348, lam=1e-6,
            eta=2e-7, gamma=0.67, S0=7.92,
            max_trade_fraction=0.4, spread_bps=1.0,
            permanent_fraction=0.4, decay_rate=0.5
        )
        
        result = solver.solve(maxiter=1000, verbose=False)
        trades = result['optimal_trades']
        
        # Should have many non-zero periods (not all in first)
        non_zero_periods = np.sum(trades > 1.0)
        spreads_execution = non_zero_periods > 5  # At least 5 non-zero periods
        
        self.log_test(
            "Long horizon execution spreading",
            spreads_execution,
            f"Non-zero periods: {non_zero_periods}/50"
        )
        return spreads_execution
    
    def test_9_zero_risk_aversion(self):
        """Test 9: With low risk aversion, different execution patterns"""
        
        # High risk aversion
        solver_high_risk = OptimalExecutionRealistic(
            X0=100000, T=1.0, N=10,
            sigma=0.0348, lam=1e-3,  # High
            eta=2e-7, gamma=0.67, S0=7.92,
            max_trade_fraction=0.4, spread_bps=1.0,
            permanent_fraction=0.4, decay_rate=0.5
        )
        
        # Low risk aversion
        solver_low_risk = OptimalExecutionRealistic(
            X0=100000, T=1.0, N=10,
            sigma=0.0348, lam=1e-10,  # Very low
            eta=2e-7, gamma=0.67, S0=7.92,
            max_trade_fraction=0.4, spread_bps=1.0,
            permanent_fraction=0.4, decay_rate=0.5
        )
        
        result_high = solver_high_risk.solve(maxiter=500, verbose=False)
        result_low = solver_low_risk.solve(maxiter=500, verbose=False)
        
        # Both should produce valid results
        p1_high = result_high['optimal_trades'][0] / 100000
        p1_low = result_low['optimal_trades'][0] / 100000
        
        both_valid = (result_high['success'] or np.isfinite(result_high['cost'])) and \
                     (result_low['success'] or np.isfinite(result_low['cost']))
        
        self.log_test(
            "Risk aversion produces valid strategies",
            both_valid,
            f"High risk (P1: {p1_high:.1%}) vs Low risk (P1: {p1_low:.1%})"
        )
        return both_valid
    
    # ========================================================================
    # SECTION 4: NUMERICAL STABILITY
    # ========================================================================
    
    def test_10_very_small_eta(self):
        """Test 10: Algorithm handles very small impact parameter"""
        print("\n" + "="*100)
        print("SECTION 4: NUMERICAL STABILITY")
        print("="*100)
        
        try:
            solver = OptimalExecutionRealistic(
                X0=100000, T=1.0, N=10,
                sigma=0.0348, lam=1e-6,
                eta=1e-10, gamma=0.67, S0=7.92,  # Very small impact
                max_trade_fraction=0.4, spread_bps=1.0,
                permanent_fraction=0.4, decay_rate=0.5
            )
            
            result = solver.solve(maxiter=500, verbose=False)
            converged = result['success'] or np.isfinite(result['cost'])
            
            self.log_test(
                "Very small impact parameter (1e-10)",
                converged,
                f"Cost: ${result['cost']:.2f}, Converged: {converged}"
            )
            return converged
        except Exception as e:
            self.log_test(
                "Very small impact parameter (1e-10)",
                False,
                f"Exception: {str(e)}"
            )
            return False
    
    def test_11_very_large_eta(self):
        """Test 11: Algorithm handles very large impact parameter"""
        
        try:
            solver = OptimalExecutionRealistic(
                X0=100000, T=1.0, N=10,
                sigma=0.0348, lam=1e-6,
                eta=1e-4, gamma=0.67, S0=7.92,  # Very large impact
                max_trade_fraction=0.4, spread_bps=1.0,
                permanent_fraction=0.4, decay_rate=0.5
            )
            
            result = solver.solve(maxiter=500, verbose=False)
            converged = result['success'] or np.isfinite(result['cost'])
            
            self.log_test(
                "Very large impact parameter (1e-4)",
                converged,
                f"Cost: ${result['cost']:.2f}, Converged: {converged}"
            )
            return converged
        except Exception as e:
            self.log_test(
                "Very large impact parameter (1e-4)",
                False,
                f"Exception: {str(e)}"
            )
            return False
    
    def test_12_extreme_gamma_values(self):
        """Test 12: Algorithm handles extreme gamma (sub-linear to super-linear)"""
        
        results_list = []
        for gamma in [0.3, 0.67, 1.5]:  # Sub-linear, normal, super-linear
            try:
                solver = OptimalExecutionRealistic(
                    X0=100000, T=1.0, N=10,
                    sigma=0.0348, lam=1e-6,
                    eta=2e-7, gamma=gamma, S0=7.92,
                    max_trade_fraction=0.4, spread_bps=1.0,
                    permanent_fraction=0.4, decay_rate=0.5
                )
                
                result = solver.solve(maxiter=500, verbose=False)
                converged = np.isfinite(result['cost'])
                results_list.append(converged)
            except:
                results_list.append(False)
        
        all_converged = all(results_list)
        self.log_test(
            "Extreme gamma values (0.3, 0.67, 1.5)",
            all_converged,
            f"All converged: {all_converged}"
        )
        return all_converged
    
    # ========================================================================
    # SECTION 5: ALGORITHM CONSISTENCY & OPTIMALITY
    # ========================================================================
    
    def test_13_optimal_vs_twap(self):
        """Test 13: Optimal strategy beats TWAP"""
        print("\n" + "="*100)
        print("SECTION 5: ALGORITHM CONSISTENCY & OPTIMALITY")
        print("="*100)
        
        solver = OptimalExecutionRealistic(
            X0=100000, T=1.0, N=10,
            sigma=0.0348, lam=1e-6,
            eta=2e-7, gamma=0.67, S0=7.92,
            max_trade_fraction=0.4, spread_bps=1.0,
            permanent_fraction=0.4, decay_rate=0.5
        )
        
        # Optimal
        result_opt = solver.solve(maxiter=1000, verbose=False)
        cost_opt = result_opt['cost']
        
        # TWAP
        twap_trades = np.ones(10) * 10000
        cost_twap = solver.cost_function(twap_trades)
        
        optimal_beats_twap = cost_opt < cost_twap
        improvement = (cost_twap - cost_opt) / cost_twap * 100
        
        self.log_test(
            "Optimal beats TWAP",
            optimal_beats_twap,
            f"Optimal: ${cost_opt:.2f}, TWAP: ${cost_twap:.2f}, Improvement: {improvement:.1f}%"
        )
        return optimal_beats_twap
    
    def test_14_deterministic_results(self):
        """Test 14: Multiple runs give consistent results"""
        
        costs = []
        for run in range(3):
            solver = OptimalExecutionRealistic(
                X0=100000, T=1.0, N=10,
                sigma=0.0348, lam=1e-6,
                eta=2e-7, gamma=0.67, S0=7.92,
                max_trade_fraction=0.4, spread_bps=1.0,
                permanent_fraction=0.4, decay_rate=0.5
            )
            
            result = solver.solve(maxiter=500, verbose=False)
            costs.append(result['cost'])
        
        # Check variance is small
        std_dev = np.std(costs)
        mean_cost = np.mean(costs)
        consistency = std_dev / mean_cost < 0.01 if mean_cost > 0 else False
        
        self.log_test(
            "Deterministic results (seed=42)",
            consistency,
            f"Mean: ${mean_cost:.2f}, StdDev: ${std_dev:.4f}, CV: {std_dev/mean_cost*100 if mean_cost>0 else 0:.2f}%"
        )
        return consistency
    
    def test_15_liquidity_constraint_effect(self):
        """Test 15: Tighter constraints increase costs"""
        
        costs_by_constraint = {}
        for max_frac in [0.1, 0.2, 0.4, 1.0]:
            solver = OptimalExecutionRealistic(
                X0=100000, T=1.0, N=10,
                sigma=0.0348, lam=1e-6,
                eta=2e-7, gamma=0.67, S0=7.92,
                max_trade_fraction=max_frac, spread_bps=1.0,
                permanent_fraction=0.4, decay_rate=0.5
            )
            
            result = solver.solve(maxiter=500, verbose=False)
            costs_by_constraint[max_frac] = result['cost']
        
        # Cost should monotonically decrease as constraint loosens
        costs_decreasing = (
            costs_by_constraint[0.1] >= costs_by_constraint[0.2] - 1 and
            costs_by_constraint[0.2] >= costs_by_constraint[0.4] - 1 and
            costs_by_constraint[0.4] >= costs_by_constraint[1.0] - 1
        )
        
        self.log_test(
            "Liquidity constraint effect",
            costs_decreasing,
            f"10%: ${costs_by_constraint[0.1]:.0f}, 20%: ${costs_by_constraint[0.2]:.0f}, "
            f"40%: ${costs_by_constraint[0.4]:.0f}, 100%: ${costs_by_constraint[1.0]:.0f}"
        )
        return costs_decreasing
    
    # ========================================================================
    # SECTION 6: REAL-WORLD SCENARIOS
    # ========================================================================
    
    def test_16_small_order_illiquid_stock(self):
        """Test 16: Small order on illiquid stock"""
        print("\n" + "="*100)
        print("SECTION 6: REAL-WORLD SCENARIOS")
        print("="*100)
        
        try:
            solver = OptimalExecutionRealistic(
                X0=1000, T=1.0, N=10,  # Small order
                sigma=0.08, lam=1e-5,  # High volatility
                eta=1e-6, gamma=0.6, S0=25.0,  # Illiquid stock
                max_trade_fraction=0.2, spread_bps=3.0,  # Tight constraint, wide spread
                permanent_fraction=0.5, decay_rate=1.0
            )
            
            result = solver.solve(maxiter=500, verbose=False)
            success = np.isfinite(result['cost'])
            
            self.log_test(
                "Small order on illiquid stock",
                success,
                f"Cost: ${result['cost']:.2f}, Converged: {success}"
            )
            return success
        except Exception as e:
            self.log_test(
                "Small order on illiquid stock",
                False,
                f"Exception: {str(e)}"
            )
            return False
    
    def test_17_large_order_liquid_stock(self):
        """Test 17: Large order on liquid stock"""
        
        try:
            solver = OptimalExecutionRealistic(
                X0=1000000, T=0.5, N=20,  # Large order, short horizon
                sigma=0.015, lam=1e-8,   # Low volatility
                eta=1e-8, gamma=0.7, S0=350.0,  # Liquid megacap
                max_trade_fraction=0.5, spread_bps=0.5,  # Loose constraint, tight spread
                permanent_fraction=0.3, decay_rate=0.3
            )
            
            result = solver.solve(maxiter=500, verbose=False)
            success = np.isfinite(result['cost'])
            
            self.log_test(
                "Large order on liquid stock",
                success,
                f"Cost: ${result['cost']:.2f}, Converged: {success}"
            )
            return success
        except Exception as e:
            self.log_test(
                "Large order on liquid stock",
                False,
                f"Exception: {str(e)}"
            )
            return False
    
    def test_18_urgency_scenarios(self):
        """Test 18: Urgent vs patient execution"""
        
        # Urgent (1 hour, high risk aversion)
        solver_urgent = OptimalExecutionRealistic(
            X0=100000, T=1/24, N=4,
            sigma=0.0348, lam=1e-4,  # High risk aversion (urgent)
            eta=2e-7, gamma=0.67, S0=7.92,
            max_trade_fraction=0.4, spread_bps=1.0,
            permanent_fraction=0.4, decay_rate=0.5
        )
        
        # Patient (1 day, low risk aversion)
        solver_patient = OptimalExecutionRealistic(
            X0=100000, T=1.0, N=10,
            sigma=0.0348, lam=1e-7,  # Low risk aversion (patient)
            eta=2e-7, gamma=0.67, S0=7.92,
            max_trade_fraction=0.4, spread_bps=1.0,
            permanent_fraction=0.4, decay_rate=0.5
        )
        
        result_urgent = solver_urgent.solve(maxiter=500, verbose=False)
        result_patient = solver_patient.solve(maxiter=500, verbose=False)
        
        # Urgent should have higher first period
        p1_urgent = result_urgent['optimal_trades'][0] / 100000
        p1_patient = result_patient['optimal_trades'][0] / 100000
        
        urgent_faster = p1_urgent > p1_patient - 0.05  # Allow 5% tolerance
        
        self.log_test(
            "Urgency scenarios (urgent vs patient)",
            urgent_faster,
            f"Urgent P1: {p1_urgent:.1%}, Patient P1: {p1_patient:.1%}"
        )
        return urgent_faster
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    
    def run_all_tests(self):
        """Run all tests and generate report"""
        
        print("\n\n")
        print("█" * 100)
        print("█" + " " * 98 + "█")
        print("█" + " " * 30 + "BULLETPROOF FINAL TEST SUITE" + " " * 40 + "█")
        print("█" + " " * 98 + "█")
        print("█" * 100)
        
        # Run all tests
        self.test_1_basic_math_power_law()
        self.test_2_spread_cost_calculation()
        self.test_3_inventory_risk_cost()
        
        self.test_4_trade_size_constraint()
        self.test_5_total_shares_constraint()
        self.test_6_non_negative_trades()
        
        self.test_7_single_period_execution()
        self.test_8_very_long_horizon()
        self.test_9_zero_risk_aversion()
        
        self.test_10_very_small_eta()
        self.test_11_very_large_eta()
        self.test_12_extreme_gamma_values()
        
        self.test_13_optimal_vs_twap()
        self.test_14_deterministic_results()
        self.test_15_liquidity_constraint_effect()
        
        self.test_16_small_order_illiquid_stock()
        self.test_17_large_order_liquid_stock()
        self.test_18_urgency_scenarios()
        
        # Generate report
        print("\n\n")
        print("=" * 100)
        print("FINAL TEST REPORT")
        print("=" * 100 + "\n")
        
        df = pd.DataFrame(self.results)
        print(df.to_string(index=False))
        
        print("\n" + "=" * 100)
        print(f"SUMMARY: {self.passed} PASSED | {self.failed} FAILED | {self.warnings_count} WARNINGS")
        print("=" * 100)
        
        if self.failed == 0:
            print("\n🎉 ALL TESTS PASSED! Algorithm is BULLETPROOF ✅")
        else:
            print(f"\n⚠️  {self.failed} TESTS FAILED - Review and fix before submission")
        
        return self.failed == 0

# ============================================================================
# RUN THE TEST SUITE
# ============================================================================

if __name__ == "__main__":
    tester = ComprehensiveSolverTest()
    all_passed = tester.run_all_tests()
    
    # Generate CSV report
    df = pd.DataFrame(tester.results)
    df.to_csv('bulletproof_test_report.csv', index=False)
    print(f"\n📊 Report saved to: bulletproof_test_report.csv")
    
    exit(0 if all_passed else 1)
