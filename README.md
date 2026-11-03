# Hedging exotics with vanillas: literature review

Papers are grouped by theme. Each entry records what the paper talks about, what might be usable for the project, and what it does not cover. It is a initial classification and review, to be complemtented as we peogress.

## Repository layout

| Path | Contents |
|---|---|
| `README.md` | this review |
| `refs.bib` | bibliographic source |

## Papers grouped by theme

### Starting point papers

1. Bel Hadj Ayed & Loeper (2023)

    *The cost of vega-hedging structured products* — [SSRN 4550643](https://papers.ssrn.com/abstract=4550643) — BNP Paribas Global Markets — `belhadjayed2023cost`

    Computes the future hedging cost of dynamically vega-hedging an autocallable note with a menu of
    vanillas under transaction costs.

    _to be read carefully_


2. Fernandez-Tapia & Guéant (2020)

    *Recipes for hedging exotics with illiquid vanillas* — [arXiv:2005.10064](https://arxiv.org/abs/2005.10064) — `fernandeztapia2020recipes`

    An exotic book is hedged with a menu of costly vanillas. The rebalancing problem is reduced to a
    Bolza problem and solved through a Hamiltonian system in $(p,q)$.

    Under a one-factor stochastic volatility model the instrument covariance has rank one, so the hedge
    reduces to a single basket and the selection problem does not arise. The exotic is valued in a
    recalibrated local volatility model while risk is measured under stochastic volatility.

3. Guéant & Pu (2017)

    *Option pricing and hedging with execution costs and market impact* — Mathematical Finance 27(3):803–831 — [arXiv:1311.4342](https://arxiv.org/abs/1311.4342) — `gueant2017option`

    Pricing and hedging when the hedge pays execution costs and moves the price. The indifference price
    solves a nonlinear PDE whose value function $\theta$ is a viscosity solution, characterised through a
    zero-sum game in which one player controls the drift, which is where permanent impact enters. 

4. Burnett (2020)

    *Hedging valuation adjustment: fact and friction* — [SSRN 3709477](https://papers.ssrn.com/abstract=3709477) — `burnett2020hedging`

    _to be read carefully_

5. Burnett & Williams (2021)

    *The cost of hedging XVA* — SSRN — `burnett2021cost`

    _to be read carefully_

    
6. Bénézet, Crépey & Essaket (2023)

    *The recalibration conundrum: hedging valuation adjustment for callable claims* — [arXiv:2304.02479](https://arxiv.org/abs/2304.02479) — revised January 2026 — `benezet2023recalibration`

    Addresses the gap between hedging theory, which assumes a single fixed model, and practice, which
    recalibrates daily. Standard model-risk reserves correct traders' valuations but neither their hedges
    nor their exercise decisions, so the authors propose a risk-adjusted reserve, illustrated on a
    stylised callable range accrual.
    

### Deep hedging and algorithms

1. Buehler, Gonon, Teichmann & Wood (2019)

    *Deep hedging* — Quantitative Finance 19(8):1271–1291 — [arXiv:1802.03042](https://arxiv.org/abs/1802.03042) — `buehler2019deep`

    The hedging policy is parametrised by a neural network, paths are simulated, and a convex risk
    measure of terminal P&L is minimised by gradient descent through the simulator. Transaction costs
    enter as terms in the objective.

    The published experiments do not model price impact: costs are proportional and exogenous, and the
    hedger's trades do not move the market.

2.  Buehler, Murray & Wood (2022)

    *Deep Bellman hedging* — [arXiv:2207.00932](https://arxiv.org/abs/2207.00932) — `buehler2022bellman`

    Actor–critic reformulation learning from historical data rather than from a simulator, which removes
    the dependence on a market model. No official public implementation found.

3. Godin (2026)

    *Is deep hedging reinforcement learning? A response to the view that Monte Carlo-based policy gradient methods fall outside the reinforcement learning framework* — [arXiv:2607.13353](https://arxiv.org/abs/2607.13353) — `godin2026deep`

    Concludes that deep hedging is a Monte Carlo, actor-only, pathwise-gradient policy-gradient RL
    algorithm, on the grounds that Sutton and Barto define RL by the problem rather than by tools such as
    value functions, Bellman equations or temporal-difference updates.

    The note accepts the factual observations of the opposing view — no value function, no Bellman
    equation, no TD updates, no explicit exploration, pathwise rather than score-function gradient — and
    disputes only whether those features exclude the method from RL. The question is one of terminology.

4. Imaki, Imajo, Ito, Minami & Nakagawa (2021)

    *No-transaction band network: a neural network architecture for efficient deep hedging* — [arXiv:2103.01775](https://arxiv.org/abs/2103.01775) — `imaki2021notransaction`

    The network outputs band endpoints $[b_l, b_u]$ from the market state alone, around the
    Black–Scholes delta, and the next position is the previous one clamped to that band. The previous
    position enters only through the clamp, so the band structure is encoded rather than learned:
    convergence in under 100 simulations against more than 1000 for a feed-forward policy.

    The conclusion lists as open the extension to several hedging instruments, the restriction to finite
    delta and gamma, and the absence of market impact.

### Deep hedging with options: the Concordia and HEC Montréal group

Much of the recent work on deep hedging with options among the hedging instruments comes from one
group: Frédéric Godin (Concordia, Department of Mathematics and Statistics, and Quantact Laboratory
at the CRM) with Alexandre Carbonneau at Concordia, and with Pascal François and Geneviève Gauthier
at HEC Montréal. The note on the RL classification in point 3 above is by the same group.

1. François, Gauthier, Godin & Pérez-Mendoza (2025)

    *Deep hedging with options using the implied volatility surface* — [arXiv:2504.06208](https://arxiv.org/abs/2504.06208) — `francois2025deep`
    Code: <https://github.com/cpmendoza/deep-hedging_with_options>

    A short ATM straddle is hedged with cash, the underlying and one longer-dated ATM call, rebalanced
    daily under proportional costs, minimising a risk measure on terminal hedging error with a soft
    constraint against doubling strategies. The simulator is the JIVR model: five surface factors with
    NGARCH–NIG dynamics, estimated on S&P 500 data 1996–2020.

    Including the hedged claim's own price, delta and gamma in the state improves every metric
    substantially. The agent rebalances far more often than delta-gamma at comparable total cost.

    The hedged claim is a vanilla; the menu holds one option at fixed moneyness; the option cost is a
    flat percentage of price, so cost per unit of vega does not depend on strike or maturity; and there
    is no market impact. The no-transaction region is an $L^1$ ball around the network's own proposed
    action, whose radius is learned to be essentially zero at every cost level.
    

2. François, Gauthier, Godin & Pérez-Mendoza (2024)

    *Is the difference between deep hedging and delta hedging a statistical arbitrage?* — [arXiv:2407.14736](https://arxiv.org/abs/2407.14736) — `francois2024statistical`

    Tests whether the difference between the deep hedging and delta hedging positions behaves as a
    statistical arbitrage in a GARCH-based incomplete market. The difference is a speculative overlay
    when the CVaR confidence level is low; a level of 85% or above prevents it.

    Relevant to the choice of confidence level if CVaR is used as the objective, and the precedent for
    the statistical-arbitrage check run in the 2025 paper.

3. Neagu, Godin, Simard & Kosseim (2024)

    *Deep hedging with market impact* — [arXiv:2402.13326](https://arxiv.org/abs/2402.13326) — Canadian AI — `neagu2024deep`

    Places market impact inside a deep hedging objective. The relevant friction for this project is impact on the vanillas rather than on the underlying.
