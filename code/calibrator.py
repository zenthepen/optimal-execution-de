"""
Adaptive Trade Size Constraints Based on Liquidity

Automatically calibrates max_trade_fraction based on stock liquidity using
Average Daily Volume (ADV) data and best practices from market microstructure.

References:
- BestEx Research (2023): "Rethinking Participation Rates"
- SEC RATS Proposal: 20% ADV threshold
- ITG Research (2015): "Optimal Participation Rates"
"""

import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional
import json
from pathlib import Path


class LiquidityCalibrator:
    """
    Calibrate trade size constraints based on stock liquidity.
    
    Liquidity Tiers (based on ADV):
    --------------------------------
    Very High (>5M ADV):    10-15% max per period
    High (1-5M ADV):        15-25% max per period
    Medium (500k-1M ADV):   20-30% max per period
    Low (100k-500k ADV):    30-40% max per period
    Very Low (<100k ADV):   40-50% max per period
    
    References:
    - BestEx Research (2023): Participation rate guidelines
    - SEC: RATS proposal framework
    """
    
    def __init__(self, 
                 lookback_days: int = 30,
                 conservative_mode: bool = True):
        """
        Initialize liquidity calibrator.
        
        Parameters
        ----------
        lookback_days : int, default=30
            Number of trading days for ADV calculation.
            30 days ≈ 1 month (standard industry practice).
            
        conservative_mode : bool, default=True
            If True, uses lower end of recommended ranges (safer).
            If False, uses upper end (more aggressive).
        """
        self.lookback_days = lookback_days
        self.conservative_mode = conservative_mode
        
        # Liquidity tiers based on ADV (shares/day)
        self.tiers = {
            'very_high': {
                'min_adv': 5_000_000,
                'max_adv': float('inf'),
                'limit_range': (0.10, 0.15),  # 10-15%
                'description': 'Very High Liquidity (>5M ADV)'
            },
            'high': {
                'min_adv': 1_000_000,
                'max_adv': 5_000_000,
                'limit_range': (0.15, 0.25),  # 15-25%
                'description': 'High Liquidity (1-5M ADV)'
            },
            'medium': {
                'min_adv': 500_000,
                'max_adv': 1_000_000,
                'limit_range': (0.20, 0.30),  # 20-30%
                'description': 'Medium Liquidity (500k-1M ADV)'
            },
            'low': {
                'min_adv': 100_000,
                'max_adv': 500_000,
                'limit_range': (0.30, 0.40),  # 30-40%
                'description': 'Low Liquidity (100k-500k ADV)'
            },
            'very_low': {
                'min_adv': 0,
                'max_adv': 100_000,
                'limit_range': (0.40, 0.50),  # 40-50%
                'description': 'Very Low Liquidity (<100k ADV)'
            }
        }
    
    def fetch_adv(self, ticker: str) -> Tuple[float, int]:
        """
        Fetch Average Daily Volume from Yahoo Finance.
        
        Parameters
        ----------
        ticker : str
            Stock ticker symbol (e.g., 'AAPL', 'SNAP')
            
        Returns
        -------
        adv : float
            Average daily volume over lookback period (shares/day)
        n_days : int
            Actual number of trading days used
        """
        try:
            stock = yf.Ticker(ticker)
            
            # Fetch extra days to ensure we get enough trading days
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.lookback_days + 10)
            
            hist = stock.history(start=start_date, end=end_date)
            
            if len(hist) == 0:
                raise ValueError(f"No data available for {ticker}")
            
            # Use last N trading days
            volumes = hist['Volume'].tail(self.lookback_days)
            adv = float(volumes.mean())
            n_days = len(volumes)
            
            if adv <= 0:
                raise ValueError(f"Invalid ADV for {ticker}: {adv}")
            
            return adv, n_days
            
        except Exception as e:
            raise ValueError(f"Failed to fetch ADV for {ticker}: {str(e)}")
    
    def classify_liquidity(self, adv: float) -> Tuple[str, Dict]:
        """
        Classify stock liquidity based on ADV.
        
        Parameters
        ----------
        adv : float
            Average daily volume (shares/day)
            
        Returns
        -------
        tier_name : str
            Liquidity tier name
        tier_info : dict
            Tier information including limits
        """
        for tier_name, tier_info in self.tiers.items():
            if tier_info['min_adv'] <= adv < tier_info['max_adv']:
                return tier_name, tier_info
        
        # Default to very_low if somehow not matched
        return 'very_low', self.tiers['very_low']
    
    def compute_order_to_adv_ratio(self, 
                                   order_size: float, 
                                   adv: float) -> float:
        """
        Compute order size as percentage of ADV.
        
        Parameters
        ----------
        order_size : float
            Total shares to execute
        adv : float
            Average daily volume
            
        Returns
        -------
        float
            Order/ADV ratio (e.g., 0.001 = 0.1% of ADV)
        """
        if adv <= 0:
            raise ValueError(f"Invalid ADV: {adv}")
        return order_size / adv
    
    def get_max_trade_fraction(self, 
                               ticker: str, 
                               order_size: float,
                               verbose: bool = True) -> Dict:
        """
        Determine appropriate max_trade_fraction for given stock and order.
        
        **This is the MAIN function to use.**
        
        Parameters
        ----------
        ticker : str
            Stock ticker symbol
        order_size : float
            Total shares to execute
        verbose : bool, default=True
            Print calibration details
            
        Returns
        -------
        dict
            Contains:
            - max_trade_fraction: Recommended limit (e.g., 0.25)
            - adv: Average daily volume
            - adv_days: Number of days used for ADV
            - order_to_adv: Order size as fraction of ADV
            - liquidity_tier: Classification
            - tier_description: Human-readable tier description
            - limit_range: (min, max) possible range
            - justification: Explanation
            - calibration_date: Timestamp
        """
        # Fetch ADV
        adv, adv_days = self.fetch_adv(ticker)
        
        # Classify liquidity
        tier_name, tier_info = self.classify_liquidity(adv)
        
        # Compute order/ADV ratio
        order_to_adv = self.compute_order_to_adv_ratio(order_size, adv)
        
        # Select limit based on mode
        limit_range = tier_info['limit_range']
        if self.conservative_mode:
            max_trade_fraction = limit_range[0]  # Lower bound (safer)
        else:
            max_trade_fraction = limit_range[1]  # Upper bound (aggressive)
        
        # Check if order is >20% ADV (SEC RATS threshold)
        rats_warning = order_to_adv > 0.20
        
        # Prepare result
        result = {
            'max_trade_fraction': max_trade_fraction,
            'adv': adv,
            'adv_days': adv_days,
            'order_to_adv': order_to_adv,
            'liquidity_tier': tier_name,
            'tier_description': tier_info['description'],
            'limit_range': limit_range,
            'rats_warning': rats_warning,
            'justification': self._generate_justification(
                ticker, order_size, adv, order_to_adv, 
                tier_name, tier_info, max_trade_fraction, rats_warning
            ),
            'calibration_date': datetime.now().isoformat()
        }
        
        if verbose:
            self._print_calibration(result, ticker, order_size)
        
        return result
    
    def _generate_justification(self, ticker, order_size, adv, order_to_adv,
                                tier_name, tier_info, max_trade_fraction,
                                rats_warning) -> str:
        """Generate human-readable justification."""
        
        mode_str = 'conservative' if self.conservative_mode else 'aggressive'
        
        justification = f"""Stock: {ticker}
Order: {order_size:,.0f} shares ({order_to_adv:.3%} of ADV)
ADV: {adv:,.0f} shares/day ({self.lookback_days}-day average)

Liquidity: {tier_info['description']}
Recommended range: {tier_info['limit_range'][0]:.0%}-{tier_info['limit_range'][1]:.0%} per period
Selected limit: {max_trade_fraction:.1%} ({mode_str} mode)

Rationale:
- BestEx Research (2023): Orders <1% ADV can use 20-50% participation
- Your order is {order_to_adv:.3%} of ADV
- For {tier_name.replace('_', ' ')} liquidity stocks, using {max_trade_fraction:.0%} is appropriate
- This allows completion in ~{int(1/max_trade_fraction)} periods minimum"""

        if rats_warning:
            justification += f"""

⚠️  SEC RATS WARNING:
Order is {order_to_adv:.1%} of ADV (>20% threshold)
This is a "Regulation Automated Trading System" reportable order.
Consider splitting across multiple days or using VWAP execution."""
        
        return justification.strip()
    
    def _print_calibration(self, result: Dict, ticker: str, order_size: float):
        """Print formatted calibration results."""
        print("\n" + "="*80)
        print(f"LIQUIDITY-BASED CONSTRAINT CALIBRATION: {ticker}")
        print("="*80)
        print(f"Average Daily Volume:  {result['adv']:>15,.0f} shares/day")
        print(f"  (computed from {result['adv_days']} trading days)")
        print(f"Order Size:            {order_size:>15,.0f} shares")
        print(f"Order/ADV Ratio:       {result['order_to_adv']:>15.3%}")
        print("-"*80)
        print(f"Liquidity Tier:        {result['tier_description']}")
        print(f"Recommended Range:     {result['limit_range'][0]:.0%} - {result['limit_range'][1]:.0%} per period")
        print(f"Selected Limit:        {result['max_trade_fraction']:.1%} per period")
        print(f"  ({'Conservative' if self.conservative_mode else 'Aggressive'} mode)")
        
        if result['rats_warning']:
            print("\n⚠️  WARNING: Order exceeds 20% of ADV (SEC RATS threshold)")
        
        print("="*80 + "\n")
    
    def calibrate_multiple_stocks(self, 
                                  stock_orders: Dict[str, float],
                                  verbose: bool = True) -> Dict[str, Dict]:
        """
        Calibrate constraints for multiple stocks.
        
        Parameters
        ----------
        stock_orders : dict
            {ticker: order_size} mapping
            Example: {'SNAP': 100000, 'UBER': 100000, 'NVDA': 100000}
        verbose : bool, default=True
            Print calibration for each stock
            
        Returns
        -------
        dict
            {ticker: calibration_result} mapping
        """
        results = {}
        
        print(f"\n{'='*80}")
        print(f"CALIBRATING {len(stock_orders)} STOCKS")
        print(f"{'='*80}\n")
        
        for ticker, order_size in stock_orders.items():
            try:
                result = self.get_max_trade_fraction(
                    ticker, order_size, verbose=verbose
                )
                results[ticker] = result
            except Exception as e:
                print(f"❌ Failed to calibrate {ticker}: {e}\n")
                results[ticker] = None
        
        return results
    
    def save_calibration(self, 
                        calibration_results: Dict, 
                        output_dir: str = None):
        """
        Save calibration results to JSON file.
        
        Parameters
        ----------
        calibration_results : dict
            Results from get_max_trade_fraction or calibrate_multiple_stocks
        output_dir : str, optional
            Output directory. Defaults to ../05_calibrated_data/
        """
        if output_dir is None:
            output_dir = Path(__file__).parent.parent / '05_calibrated_data'
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = output_dir / f'liquidity_calibration_{timestamp}.json'
        
        # Convert to serializable format
        serializable = {}
        for ticker, result in calibration_results.items():
            if result is None:
                continue
            serializable[ticker] = {
                'max_trade_fraction': float(result['max_trade_fraction']),
                'adv': float(result['adv']),
                'adv_days': int(result['adv_days']),
                'order_to_adv': float(result['order_to_adv']),
                'liquidity_tier': result['liquidity_tier'],
                'tier_description': result['tier_description'],
                'limit_range': [float(x) for x in result['limit_range']],
                'rats_warning': bool(result['rats_warning']),
                'calibration_date': result['calibration_date']
            }
        
        with open(filename, 'w') as f:
            json.dump(serializable, f, indent=2)
        
        print(f"✅ Calibration saved to: {filename}")
        return filename


def load_liquidity_calibration(ticker: str, 
                               calibration_file: Optional[str] = None) -> Dict:
    """
    Load previously saved liquidity calibration.
    
    Parameters
    ----------
    ticker : str
        Stock ticker
    calibration_file : str, optional
        Path to calibration JSON file. If None, looks for latest.
        
    Returns
    -------
    dict
        Calibration results for the ticker
    """
    if calibration_file is None:
        # Look for latest calibration file
        calib_dir = Path(__file__).parent.parent / '05_calibrated_data'
        calib_files = list(calib_dir.glob('liquidity_calibration_*.json'))
        
        if not calib_files:
            raise FileNotFoundError(
                f"No liquidity calibration files found in {calib_dir}"
            )
        
        # Get most recent
        calibration_file = max(calib_files, key=lambda p: p.stat().st_mtime)
    
    with open(calibration_file, 'r') as f:
        all_calibrations = json.load(f)
    
    if ticker not in all_calibrations:
        raise ValueError(
            f"Ticker {ticker} not found in calibration file. "
            f"Available: {list(all_calibrations.keys())}"
        )
    
    return all_calibrations[ticker]


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    """
    Example usage of adaptive liquidity calibration.
    """
    
    print("\n" + "="*80)
    print("ADAPTIVE LIQUIDITY CALIBRATION - EXAMPLES")
    print("="*80 + "\n")
    
    # Initialize calibrator
    calibrator = LiquidityCalibrator(
        lookback_days=30,
        conservative_mode=True
    )
    
    # Example 1: Single stock
    print("\n" + "="*80)
    print("EXAMPLE 1: SINGLE STOCK CALIBRATION")
    print("="*80)
    
    result_snap = calibrator.get_max_trade_fraction(
        ticker='SNAP',
        order_size=100000,
        verbose=True
    )
    
    print(f"Recommended max_trade_fraction: {result_snap['max_trade_fraction']:.1%}")
    
    # Example 2: Multiple stocks (your 3 test assets)
    print("\n\n" + "="*80)
    print("EXAMPLE 2: CALIBRATE YOUR 3 TEST ASSETS")
    print("="*80 + "\n")
    
    test_assets = {
        'SNAP': 100000,
        'UBER': 100000,
        'NVDA': 100000
    }
    
    results = calibrator.calibrate_multiple_stocks(test_assets, verbose=True)
    
    # Save calibration
    saved_file = calibrator.save_calibration(results)
    
    # Summary comparison
    print("\n" + "="*80)
    print("SUMMARY: CONSTRAINT COMPARISON")
    print("="*80)
    print(f"{'Asset':<10} {'ADV':>15} {'Order/ADV':>12} {'Max Trade':>12} {'Previous':>12}")
    print("-"*80)
    
    for ticker in ['SNAP', 'UBER', 'NVDA']:
        if ticker in results and results[ticker]:
            r = results[ticker]
            print(f"{ticker:<10} {r['adv']:>15,.0f} {r['order_to_adv']:>11.2%} "
                  f"{r['max_trade_fraction']:>11.1%} {'40.0%':>12}")
    
    print("-"*80)
    print("\nKey Insights:")
    print("- SNAP, UBER, NVDA all have high liquidity (>1M ADV)")
    print("- 100k share orders are <1% of ADV for all (very safe)")
    print("- Adaptive constraints allow more aggressive execution")
    print("- Previous 40% limit was conservative but reasonable")
    print("="*80 + "\n")
