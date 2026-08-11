# Temporal Difference Learning & Control Theory Interpretations

This repository contains materials for an academic research project focused on studying **Reinforcement Learning (RL)** through the perspective of **control theory, dynamical systems, and feedback control principles**.

The project implements foundational temporal-difference algorithms (**TD(0)**, **Monte Carlo**, **Q-learning**, and **SARSA**) and provides a numerical analysis of their behavior across three classic environments: *Random Walk*, *Gridworld*, and *Pendulum-v1*.

## Theoretical Foundations

### 1. Markov Decision Process (MDP)

The dynamics of the decision process are formalized as a tuple $\langle \mathcal{S}, \mathcal{A}, P, R, \gamma \rangle$:

* $\mathcal{S}$ represents the state space, and $\mathcal{A}$ represents the action space.
* $P_{ss'}^a = \mathbb{P}(S_{t+1}=s' \mid S_t=s, A_t=a)$ denotes the transition probability.
* $R_s^a = \mathbb{E}[R_{t+1} \mid S_t=s, A_t=a]$ is the reward function.
* $\gamma \in [0, 1)$ is the discount factor.

State-value $V^\pi(s)$ and action-value $Q^\pi(s,a)$ functions under policy $\pi$:

$$V^\pi(s) = \mathbb{E}_\pi \left[ \sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \;\Bigg|\; S_t = s \right]$$

$$Q^\pi(s,a) = \mathbb{E}_\pi \left[ \sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \;\Bigg|\; S_t = s, A_t = a \right]$$

### 2. Bellman Operators & Banach Fixed-Point Theorem

The linear Bellman operator for a fixed policy $\pi$:

$$T_\pi(v) = R^\pi + \gamma P^\pi v$$

The Bellman optimality operator:

$$(T_* v)(s) = \max_{a \in \mathcal{A}} \left( R_s^a + \gamma \sum_{s' \in \mathcal{S}} P_{ss'}^a v(s') \right)$$

#### Contraction Proofs for $T_\pi$ and $T_*$

In the complete metric space $\mathbb{R}^{|\mathcal{S}|}$ equipped with the maximum norm $\|v\|_\infty = \max_{s \in \mathcal{S}} |v(s)|$:

1. **For $T_\pi$:**

   $$\|T_\pi u - T_\pi v\|_\infty = \|\gamma P^\pi (u - v)\|_\infty \le \gamma \|P^\pi\|_\infty \|u - v\|_\infty = \gamma \|u - v\|_\infty$$

   Since $P^\pi$ is a stochastic matrix ($\|P^\pi\|_\infty = 1$) and $\gamma < 1$, $T_\pi$ is a contraction mapping.

2. **For $T_*$:**

   For any state $s \in \mathcal{S}$, assuming the maximum for $(T_* u)(s)$ is attained at action $a_1$:

   $$(T_* u)(s) - (T_* v)(s) \le \gamma \sum_{s'} P_{ss'}^{a_1} (u(s') - v(s')) \le \gamma \|u - v\|_\infty \sum_{s'} P_{ss'}^{a_1} = \gamma \|u - v\|_\infty$$

   Evaluating $(T_* v)(s) - (T_* u)(s)$ similarly yields:

   $$\|T_* u - T_* v\|_\infty \le \gamma \|u - v\|_\infty$$

*By Banach's fixed-point theorem, there exists a unique fixed point $v_\pi = T_\pi v_\pi$ and $v_* = T_* v_*$ to which iterative evaluations are guaranteed to converge.*

### 3. RL in Terms of Control Theory & Dynamical Systems

* **Error Signal / Feedback:** The TD-error $\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$ mirrors the system tracking error $e(t) = u(t) - y(t)$ in classical closed-loop control systems.
* **Control Synthesis:** Greedy action selection over $Q$-values corresponds to analytical controller synthesis, where $V(s)$ or $Q(s,a)$ acts as a discrete Lyapunov function.
* **Two-Time-Scale Dynamics:** Fast time-scale dynamics govern environment state transitions $S_t \to S_{t+1}$, whereas slow time-scale dynamics govern value parameter updates ($Q$/$V$).

---

## Implemented Experiments & Environments

### 1. Random Walk

* **Description:** A 7-state discrete Markov chain $\{0, 1, \dots, 6\}$ with terminal states $0$ and $6$.
* **Objective:** Evaluating state-value $V(s)$ estimation accuracy between **TD(0)** and **Constant-$\alpha$ Monte Carlo**.
* **Findings:** TD(0) exhibits substantially lower RMSE and MAE curves due to reduced variance in sample step updates achieved via bootstrapping.

### 2. Gridworld (5x5 Grid)

* **Description:** A $5 \times 5$ environment starting at $(0,0)$ with goal state $(4,4)$. Step reward is $-1$, with $+10$ awarded upon reaching the goal.
* **Objective:** Analyzing the impact of hyperparameters $\alpha$, $\gamma$, and the $\epsilon$-greedy exploration policy on **Q-learning** convergence.
* **Findings:** Optimal convergence stability is achieved at $\alpha = 0.1$, $\gamma = 0.95$, $\epsilon = 0.1$. High exploration ($\epsilon = 0.5$) causes persistent reward fluctuations, while undersized exploration ($\epsilon = 0.01$) increases the risk of getting trapped in local optima.

### 3. Pendulum-v1 (Discretized Inverted Pendulum)

* **Description:** Continuous inverted pendulum stabilization task, discretized into $8 \times 8 \times 8 = 512$ states and $5$ torque control action levels $a \in \{-2.0, -1.0, 0.0, 1.0, 2.0\}$.
* **Objective:** Comparative analysis of On-policy (**SARSA**) versus Off-policy (**Q-learning**) control strategies.
* **Findings:** **Q-learning** reaches a higher return plateau ($\sim -600$) by computing updates via the maximum operator $\max_a Q(s',a)$, resulting in an aggressive, time-optimal swing-up control policy. **SARSA** converges to a lower plateau ($\sim -700$), accounting for actual exploratory actions ($\epsilon = 0.1$) to produce a conservative control policy that mitigates fall risks under stochasticity.
