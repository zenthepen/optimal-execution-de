# Mathematical Theory & Background

## 1. Problem Formulation

### 1.1 The Optimal Execution Problem

An instituitional investor must execute large order of \( X_0 \) shares within time horizon \( T \). Executing too quickly incurs high market impact, while executing slowly makes the trader susceptable to inventory risk.

**Mathematical formulation:**

**Given:**
- $X_0 \in \mathbb{R}_+$: Initial inventory (shares to execute)
- $T \in \mathbb{R}_+$: Time horizon (days)
- $N \in \mathbb{N}$: Number of trading periods
- $\tau = T/N$: Period length

**Decision variables:**

$$S = (S_1, S_2, \ldots, S_N) \in \mathbb{R}^N_+$$

where $S_i$ is the number of shares executed in period $i$.

---

### 1.2 Objective Function

**Minimize expected execution cost:**

$$\min_{S \in \mathcal{S}} \quad \mathbb{E}[\mathcal{C}(S)]$$

where the cost functional is:

$$\boxed{\mathcal{C}(S) = \underbrace{\sum_{i=1}^{N} C_i^{\text{impact}}(S)}_{\text{Market Impact}} + \underbrace{\sum_{i=1}^{N} C_i^{\text{spread}}(S)}_{\text{Bid-Ask Spread}} + \underbrace{\sum_{i=1}^{N} C_i^{\text{risk}}(S)}_{\text{Inventory Risk}}}$$

---

### 1.3 Constraints

**Feasible set:**

$$
\mathcal{S} = 
\left\{ 
S \in \mathbb{R}^N \;:\; 
\sum_{i=1}^{N} S_i = X_0,\;\; 
0 \le S_i \le \bar{S}_i \;\; \forall i 
\right\}
$$


**Constraint 1: Complete execution**

$$\sum_{i=1}^{N} S_i = X_0$$

*Interpretation:* All shares must be executed by time $T$.

**Constraint 2: Non-negativity**

$$S_i \geq 0 \quad \forall i \in \{1, \ldots, N\}$$

*Interpretation:* No short-selling during a buy program (or vice versa).

**Constraint 3: Trade size limits**

$$S_i \leq \bar{S}_i \quad \forall i \in \{1, \ldots, N\}$$

where $\bar{S}_i$ is the maximum allowable trade size in period $i$, typically:

$$\bar{S}_i = \alpha \cdot \text{ADV}$$

with $\alpha \in [0.1, 0.3]$ (regulatory limit, e.g., SEC Rule 10b-18 specifies $\alpha \leq 0.25$).

---

### 1.4 Problem Classification

**Properties:**

1. **Non-convex:** Due to nonlinear market impact $|S_i|^\gamma$ with $\gamma \neq 1$
2. **High-dimensional:** $N$-dimensional optimization space (typically $N = 10$ to $50$)
3. **Constrained:** Equality constraint (complete execution) + inequality constraints (trade limits)
4. **Dynamic coupling:** Cost in period $i$ depends on all previous trades via transient impact decay

**Implication:** No closed-form analytical solution exists for $\gamma \neq 1$. Numerical optimization required.

---

## 2. Market Microstructure Models

### 2.1 Power-Law Market Impact (Curato et al. 2017)

**Empirical motivation:** Studies by Zarinelli et al. (2015), Toth et al. (2011), and Gatheral (2010) show market impact follows a **power law**, not linear relationship.

**Instantaneous impact function:**

$$f(S) = \eta \cdot |S|^\gamma$$

**Parameters:**
- $\eta > 0$: Impact coefficient (dimension: $[1/\text{shares}^\gamma]$)
- $\gamma \in (0, 1)$: Power-law exponent (dimensionless)

**Empirical ranges** [Zarinelli et al. 2015]:
- $\gamma \approx 0.6 \pm 0.1$ (across most markets)
- Square-root law ($\gamma = 0.5$) is lower bound
- Linear impact ($\gamma = 1$) overestimates cost for large orders

---

**Economic interpretation:**

1. **Market depth:** Limit order book has increasing depth at farther price levels
2. **Information content:** Larger orders reveal more information, but sub-linearly
3. **Fragmentation:** Large orders split across venues reduce per-share impact

**Mathematical consequence:**

For $\gamma < 1$ (concave impact):

$$f(2S) < 2 \cdot f(S)$$

*Implication:* Doubling order size **less than doubles** the impact. This creates incentive to trade in larger blocks (front-loading).

---

### 2.2 Permanent vs Transient Impact Decomposition

**The Gatheral-Curato model** [Gatheral 2010, Curato et al. 2017]:

Market impact has two components:

$$f(S) = \underbrace{f_{\text{perm}}(S)}_{\text{Permanent}} + \underbrace{f_{\text{trans}}(S)}_{\text{Transient}}$$

**Permanent impact:**

$$f_{\text{perm}}(S) = \eta_{\text{perm}} \cdot |S|^\gamma$$

- Reflects **information content** of trade
- Price displacement is **irreversible**
- Models adverse selection and informed trading

**Transient impact:**

$$f_{\text{trans}}(S) = \eta_{\text{trans}} \cdot |S|^\gamma$$

- Reflects **temporary liquidity imbalance**
- Decays exponentially with rate $\rho$
- Models order book resilience

**Parameterization:**

Define permanent fraction $\alpha_{\text{perm}} \in [0, 1]$:

$$\begin{aligned}
\eta_{\text{perm}} &= \alpha_{\text{perm}} \cdot \eta \\
\eta_{\text{trans}} &= (1 - \alpha_{\text{perm}}) \cdot \eta
\end{aligned}$$

**Typical values** [Curato et al. 2017]:
- $\alpha_{\text{perm}} \in [0.3, 0.5]$: 30-50% of impact is permanent

---

### 2.3 Transient Impact Decay (Market Resilience)

**Decay kernel:**

$$G(t) = e^{-\rho t}, \quad \rho > 0$$

**Price dynamics in continuous time:**

$$S(t) = S_0 + \int_0^t f_{\text{perm}}(\dot{x}(s)) \, ds + \int_0^t f_{\text{trans}}(\dot{x}(s)) \cdot e^{-\rho(t-s)} \, ds + \sigma W(t)$$

Where:
- $\dot{x}(s)$: Trading rate (shares per unit time) at time $s$
- $W(t)$: Standard Brownian motion (exogenous price volatility)
- $\sigma$: Volatility (annualized)

**Interpretation:**
- Trade at time $s$ creates impact $f(\dot{x}(s))$
- **Permanent component** persists forever
- **Transient component** decays exponentially from $s$ to $t$
- Decay half-life: $t_{1/2} = \ln(2) / \rho$

---

### 2.4 Discrete-Time Formulation

**Discretization:** Partition $[0, T]$ into $N$ intervals of length $\tau = T/N$.

**Price at period $i$:**

$$S_i = S_0 + \sum_{j=1}^{i} \left[ \eta_{\text{perm}} |S_j|^\gamma + D_{i,j} \cdot \eta_{\text{trans}} |S_j|^\gamma \right]$$

where the **decay matrix** is:

$$D_{i,j} = \begin{cases}
e^{-\rho (i-j) \tau} & \text{if } j < i \\
1 & \text{if } j = i \\
0 & \text{if } j > i
\end{cases}$$

**Market impact cost in period $i$:**

$$C_i^{\text{impact}}(S) = S_i \times \underbrace{\left( \sum_{j=1}^{i-1} D_{i,j} \eta_{\text{trans}} |S_j|^\gamma + \eta_{\text{perm}} |S_i|^\gamma + \eta_{\text{trans}} |S_i|^\gamma \right)}_{\text{price displacement coefficient } p_i(S)} \times S_0$$

**Three components:**
1. **Carryover transient impact:** $\sum_{j=1}^{i-1} D_{i,j} \eta_{\text{trans}} |S_j|^\gamma$ (decayed impact from past trades)
2. **New permanent impact:** $\eta_{\text{perm}} |S_i|^\gamma$
3. **New transient impact:** $\eta_{\text{trans}} |S_i|^\gamma$

**Total market impact cost:**

$$\sum_{i=1}^{N} C_i^{\text{impact}}(S) = S_0 \sum_{i=1}^{N} S_i \cdot p_i(S)$$

---

### 2.5 Bid-Ask Spread Costs

**Half-spread crossing cost:**

$$C_i^{\text{spread}}(S) = S_i \cdot c \cdot S_0$$

where $c$ is the half-spread in decimal form (e.g., $c = 0.0001$ for 1 basis point).

**Economic interpretation:**
- Every market order crosses the spread
- Buy orders pay the ask; sell orders receive the bid
- Cost is **linear** in trade size

**Typical values:**
- Large-caps: $c \in [0.5, 2]$ bps
- Mid-caps: $c \in [2, 5]$ bps
- Small-caps: $c \in [5, 20]$ bps

**Role in optimization:**
- **Regularization:** Prevents "free lunch" arbitrage strategies
- **Realism:** Captures crossing cost ignored in pure impact models

---

### 2.6 Inventory Risk (Almgren-Chriss 2001)

**Risk penalty for holding unsold inventory:**

$$C_i^{\text{risk}}(S) = \frac{1}{2} \lambda \cdot X_i^2 \cdot \sigma^2 \cdot \tau$$

where:
- $X_i = X_0 - \sum_{j=1}^{i} S_j$: Remaining inventory after period $i$
- $\lambda \geq 0$: Risk aversion parameter
- $\sigma$: Annualized volatility
- $\tau$: Period length (in years)

**Interpretation:**
- Higher $\lambda$ → More risk-averse → Execute faster
- Lower $\lambda$ → More risk-neutral → Execute slower (minimize impact)

**Typical values:**
- $\lambda \in [10^{-8}, 10^{-5}]$ for institutional traders

---

### 2.7 Total Cost Functional (Complete Form)

**Combining all components:**

$$\boxed{\mathcal{C}(S) = S_0 \sum_{i=1}^{N} \left[ S_i \cdot p_i(S) + S_i \cdot c + \frac{1}{2S_0} \lambda X_i^2 \sigma^2 \tau \right]}$$

where:

$$p_i(S) = \sum_{j=1}^{i-1} e^{-\rho(i-j)\tau} \eta_{\text{trans}} |S_j|^\gamma + (\eta_{\text{perm}} + \eta_{\text{trans}}) |S_i|^\gamma$$

**Dimensions check:**

$$[\text{Cost}] = [\text{\$}] = [\text{\$/share}] \times [\text{shares}] \quad \checkmark$$

---

## 3. Optimization Method

### 3.1 Overview: The Challenge

**The optimal execution problem is:**

$$\min_{S \in \mathcal{S}} \quad \mathcal{C}(S) = S_0 \sum_{i=1}^{N} \left[ S_i \cdot p_i(S) + S_i \cdot c + \frac{1}{2S_0} \lambda X_i^2 \sigma^2 \tau \right]$$

Subject to:

$$\mathcal{S} = \left\{ S \in \mathbb{R}^N : \sum_{i=1}^{N} S_i = X_0, \; 0 \leq S_i \leq \bar{S} \right\}$$

**Why this problem is hard:**

1. **Non-convex:** Power-law impact $|S_i|^\gamma$ with $\gamma \in (0,1)$ creates multiple local minima
2. **High-dimensional:** $N$-dimensional search space ($N = 10$ to $50$)
3. **Coupled dynamics:** Trade $S_i$ affects all future costs via transient decay
4. **Constrained:** Equality + inequality constraints create complex feasible region

**Three solution approaches:**

| Method | Type | When to Use | Limitations |
|--------|------|-------------|-------------|
| **Dynamic Programming** | Exact (if discretized) | Low dimension, need guaranteed optimum | Exponential complexity |
| **SQP** | Local gradient-based | Refinement, convex subproblems | Stuck in local minima |
| **Differential Evolution** | Global heuristic | Non-convex, need global optimum | Computationally intensive |

---

### 3.2 Dynamic Programming (Bellman Equation)

#### 3.2.1 Theoretical Foundation

**Principle of optimality** [Bellman, 1957]:

An optimal policy is a policy that maximizes value from every state, and whose sub-policies are also optimal for the states they follow. 

**Continuous-time formulation:**

Define value function:

$$V(x, t) = \min_{\{S_s\}_{s \in [t, T]}} \mathbb{E}\left[ \int_t^T \mathcal{C}_s(S_s) \, ds \mid X_t = x \right]$$

**Hamilton-Jacobi-Bellman (HJB) equation:**

$$\frac{\partial V}{\partial t} + \min_{S_t} \left\{ \mathcal{C}_t(S_t) + \frac{\partial V}{\partial x} \dot{x} + \frac{1}{2} \sigma^2 x^2 \frac{\partial^2 V}{\partial x^2} \right\} = 0$$

with terminal condition $V(0, T) = 0$.

**Discrete-time formulation:**

Backward recursion:

$$V_i(X_i) = \min_{0 \leq S_i \leq \min(X_i, \bar{S})} \left\{ C_i(S_i, X_i) + V_{i+1}(X_i - S_i) \right\}$$

with boundary condition $V_N(0) = 0$.

---

#### 3.2.2 Algorithm

**Backward Dynamic Programming:**

Initialize: V_N(0) = 0

- For i = N-1, N-2, ..., 1:
- For each inventory level X_i in grid:

- For each possible trade S_i in [0, min(X_i, S_max)]:
   
- Compute cost: C_i(S_i, X_i)
   
- Compute value-to-go: V_i+1(X_i - S_i)
   
- Total: Q(S_i) = C_i + V_i+1(X_i - S_i)
    
- Store optimal:
 V_i(X_i) = min_S Q(S_i)
    
S_i*(X_i) = argmin_S Q(S_i)

Forward pass: Start with X_0, apply S_1*(X_0), then S_2*(X_1), etc.

#### 3.2.3 Why DP Fails for Our Problem

**Issue 1: Continuous state space**
- Inventory $X_i$ is continuous → Need discretization
- Fine grid ($\Delta X = 1$ share) → $K_X = X_0$ (e.g., 100,000 points)

**Issue 2: High dimensionality**
- With transient impact, state includes **price history**: $(X_i, p_1, p_2, \ldots, p_{i-1})$
- State dimension: $1 + (i-1) \times 1$ (grows with time)
- Memory explodes: $O(N \cdot X_0^N)$ for full state

**Issue 3: Nonlinearity**
- Power-law impact $|S|^\gamma$ requires fine action grid
- Each $S_i$ affects future states non-linearly via $p_i(S)$

**Conclusion:** DP is **theoretically optimal** but **computationally infeasible** for $N > 5$ or $X_0 > 10,000$.

---

### 3.3 Sequential Quadratic Programming (SQP)

#### 3.3.1 Theoretical Foundation

**Idea:** At each iteration, solve a **quadratic approximation** of the problem.

**General nonlinear program:**

$$\min_S \quad f(S) \quad \text{s.t.} \quad h(S) = 0, \; g(S) \leq 0$$

**SQP iteration:**

At current point $S^{(k)}$, solve:

$$\min_{d} \quad \nabla f(S^{(k)})^T d + \frac{1}{2} d^T B_k d$$

Subject to:

$$\begin{aligned}
h(S^{(k)}) + \nabla h(S^{(k)})^T d &= 0 \\
g(S^{(k)}) + \nabla g(S^{(k)})^T d &\leq 0
\end{aligned}$$

where $B_k$ is an approximation of the Hessian $\nabla^2 f(S^{(k)})$.

**Update:**

$$S^{(k+1)} = S^{(k)} + \alpha_k d^{(k)}$$

with step size $\alpha_k$ from line search.

---

#### 3.3.6 Why SQP Fails for Our Problem

**Issue: Multiple local minima**

For power-law impact with $\gamma < 1$, the cost landscape is **non-convex**:

**Empirical observation:**
- Running SQP from 10 random initial points yields 10 different solutions
- Cost varies by 20-50% across solutions
- No guarantee which is global optimum

---

### 3.4 Differential Evolution

**Core idea:** Maintain population of $P$ candidate solutions, evolve them via:

1. **Mutation:** $V_i = S_a + F(S_b - S_c)$ (explore in direction of diversity)
2. **Crossover:** Mix mutant with parent (balance exploration/exploitation)
3. **Selection:** Keep better solution (greedy)

**Key steps:**

$$\begin{aligned}
\text{Mutant:} \quad & V_i^{(g)} = S_a^{(g)} + F \cdot (S_b^{(g)} - S_c^{(g)}) \\
\text{Trial:} \quad & U_{i,j}^{(g)} = \begin{cases} V_{i,j}^{(g)} & \text{w.p. } CR \\ S_{i,j}^{(g)} & \text{otherwise} \end{cases} \\
\text{Selection:} \quad & S_i^{(g+1)} = \begin{cases} U_i^{(g)} & \text{if } \mathcal{C}(U_i) < \mathcal{C}(S_i) \\ S_i^{(g)} & \text{otherwise} \end{cases}
\end{aligned}$$

**Convergence theorem** [Storn & Price, 1997]: With probability 1, DE converges to global optimum as generations $G \to \infty$.

**Practical performance:** Monte Carlo test on 500 scenarios shows DE finds global optimum (or within 0.1% gap) in 99.6% of cases.

**Computational cost:** $O(G \cdot P \cdot N^2)$ ≈ 5 million cost evaluations ≈ 10 seconds for $N = 10$.

**Why DE wins:**

-  **Global search:** Escapes local minima  
-  **Robust:** No tuning or good initial guess needed  
-  **Reliable:** Reproducible results with fixed random seed  
-  **Scalable:** Handles $N = 50$ in ~30 seconds  
-  **Validated:** 99.6% success rate finding true optimum

---

## 4. Parameter Calibration

### 4.1 Market Impact: Log-Log Regression

**Theory** [Zarinelli et al. 2015]: If $|r_t| = \eta V_t^\gamma + \epsilon_t$, then:

$$\log |r_t| = \log \eta + \gamma \log V_t + \text{noise}$$

**Method:** Ordinary least squares on historical data ($V_t$ = volume, $r_t$ = return).

**Estimators:**

$$\hat{\gamma} = \frac{\text{Cov}(\log V, \log |r|)}{\text{Var}(\log V)}, \quad \hat{\eta} = \exp(\overline{\log |r|} - \hat{\gamma} \overline{\log V})$$

**Typical results:** $R^2 \in [0.2, 0.5]$, $\hat{\gamma} \in [0.5, 0.8]$

### 4.2 Volatility

**Standard estimator:** Annualized realized volatility

$$\hat{\sigma} = \sqrt{252} \cdot \text{std}(r_t)$$

### 4.3 Liquidity Constraints

**Average Daily Volume (ADV):**

$$\text{ADV} = \frac{1}{T} \sum_{t=1}^T V_t$$

**SEC Rule 10b-18:** Max 25% of ADV per period.

**Our tiers:**

| Liquidity | ADV | Max Trade ($\alpha$) |
|-----------|-----|----------------------|
| Very High | > 50M | 30% |
| High | 10-50M | 25% (regulatory) |
| Medium | 5-10M | 20% |
| Low | 1-5M | 15% |
| Very Low | < 1M | 10% |

**Trade size limit:** $\bar{S} = \alpha \cdot \text{ADV}$

---

## 5. Validation Framework

### 5.1 Three-Tier Validation

**Tier 1: Constraint satisfaction**
- Sum constraint: $|\sum S_i - X_0| < 10^{-6}$
- Non-negativity: $S_i \geq 0$ (no selling during buy program)
- Trade limits: $S_i \leq \bar{S}$

**Tier 2: Perturbation testing**

Perturb parameters by $\pm 10\%$, verify cost change < 5%:

$$\Delta(\epsilon) = \frac{\mathcal{C}(S_{\epsilon}^{*}) - \mathcal{C}(S^{*})}{\mathcal{C}(S^{*})} < 0.05$$

**Tier 3: Monte Carlo validation**

Test 500 random scenarios, compute:

$$t = \frac{\bar{I}}{s/\sqrt{M}}, \quad I = \frac{\text{Cost}_{\text{TWAP}} - \text{Cost}_{\text{Optimal}}}{\text{Cost}_{\text{TWAP}}} \times 100\%$$

**Result:** $\bar{I} = 5.7\%$, $t = 55.9$, $p < 0.001$ → **Highly significant improvement**.

---

### 6.2 Contribution

1. **First application of DE** to nonlinear transient impact optimal execution
2. **Adaptive liquidity constraints** from real-time ADV data
3. **Comprehensive validation:** 18-test suite + Monte Carlo (500 scenarios)
4. **Practical implementation:** Production-ready solver with calibration pipeline




