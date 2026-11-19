"""Small crowds, solved exactly: when does correlation beat Nash, and when does privacy matter?

In a crowd of a few dozen people one person's move still changes the others' costs, so the
large-crowd results (Proposition 2: advice alone cannot beat Nash; Section 6: private advice
adds almost nothing over a public hint) need not hold. Here the crowd is small enough to list
every schedule, so every number is exact rather than a search over a menu.

Why listing schedules is enough. People of one type are interchangeable. If some distribution
over who-goes-where is a (Bayes) correlated equilibrium, so is its average over all relabellings
of same-type people, at the same expected cost. That average is "draw a schedule X, then shuffle
inside each type", which is exactly what the schedule LP optimises. So the LP over all schedules
gives the best CE of the whole game (checked against the full normal-form LP in the tests).

What is computed:
    exact_ce               best CE and coarse CE, every pure Nash equilibrium, the optimum
    symmetric_mixed_nash   the equilibrium of identical people who cannot coordinate at all
    small_crowd_information  bad days in a small crowd: tell nobody, announce, best public hint,
                           private advice (best Bayes CE), planner -- all exact
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog
from scipy.special import gammaln

from .game import FluidQueue, SlotGame


# ----------------------------------------------------------------- listing schedules
def compositions(n: int, T: int):
    """All ways to put n identical people into T slots (tuples of length T summing to n)."""
    if T == 1:
        yield (n,)
        return
    for a in range(n + 1):
        for rest in compositions(n - a, T - 1):
            yield (a, *rest)


def all_schedules(game: SlotGame, limit: int = 200_000) -> list[np.ndarray]:
    """Every schedule X of the game (K x T, row k sums to the type-k count)."""
    size = 1
    for c in game.counts:
        size *= math.comb(int(c) + game.T - 1, game.T - 1)
    if size > limit:
        raise ValueError(f"{size:,} schedules is too many to enumerate (limit {limit:,})")
    per_type = [list(compositions(int(c), game.T)) for c in game.counts]
    return [np.array(rows, dtype=int) for rows in itertools.product(*per_type)]


def small_crowd_game(n: int, load: float = 0.8, T: int = 3, beta: float = 2.0, slot_minutes: float = 10.0
                     ) -> SlotGame:
    """n identical people who all prefer the middle slot; the bottleneck scales with the crowd.

    ``load`` = people / (people the bottleneck can serve over the whole window), so the game looks
    the same at every size and only the weight of one person changes.
    """
    mu = n / (load * T * slot_minutes)
    return SlotGame(FluidQueue(mu, T=T, slot_minutes=slot_minutes), tau=[T // 2], beta=[beta], counts=[n],
                    name=f"small crowd n={n}")


# ----------------------------------------------------------------- LP building blocks
def _columns(game: SlotGame, schedules: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Obedience coefficients and social cost of every schedule.

    G[j, r] = X_j[k,t] (C_j[k,t] - D_j[k,t,t']) for each row r = (k, t, t' != t): the weighted gain
    from disobeying. Both G and the cost are linear in the congestion model, so the columns of a
    mixture of states are the same mixture of the columns.
    """
    T = game.T
    t_idx = np.arange(T)
    off = ~np.eye(T, dtype=bool)
    G, cost = [], []
    for X in schedules:
        D = game.deviation_costs(X)
        C = D[:, t_idx, t_idx]
        G.append((X[:, :, None] * (C[:, :, None] - D))[:, off].ravel())
        cost.append(game.social_cost(X))
    return np.array(G), np.array(cost)


def _best_ce_lp(G: np.ndarray, cost: np.ndarray, coarse_rows: np.ndarray | None = None) -> tuple[float, np.ndarray]:
    A = G.T if coarse_rows is None else coarse_rows
    J = len(cost)
    res = linprog(cost, A_ub=A, b_ub=np.zeros(A.shape[0]), A_eq=np.ones((1, J)), b_eq=[1.0],
                  bounds=[(0, None)] * J, method="highs")
    if not res.success:
        raise RuntimeError(f"CE LP failed: {res.message}")
    return float(res.fun), np.clip(res.x, 0, None)


def _coarse_rows(game: SlotGame, schedules: list[np.ndarray]) -> np.ndarray:
    """CCE rows: for each type k and fixed alternative t', sum over the told slot t."""
    T = game.T
    t_idx = np.arange(T)
    cols = []
    for X in schedules:
        D = game.deviation_costs(X)
        C = D[:, t_idx, t_idx]
        cols.append((X[:, :, None] * (C[:, :, None] - D)).sum(axis=1).ravel())
    return np.array(cols).T


# ----------------------------------------------------------------- symmetric mixed Nash
@dataclass
class MixedNash:
    sigma: np.ndarray  # probability of each slot
    cost: float  # expected social cost (total over the crowd)
    gap: float  # largest gain (minutes) a person could get by switching to a pure slot


def symmetric_mixed_nash(game: SlotGame, starts: int = 6, iters: int = 20_000, seed: int = 0) -> list[MixedNash]:
    """Symmetric mixed equilibria of a one-type game: everyone draws a slot from the same sigma.

    This is what identical people do when they cannot coordinate at all (the "no information"
    outcome of the two-student game). A person choosing t meets A = e_t + M with M multinomial
    (n-1, sigma); we list every M exactly and find sigma with exponential-weights dynamics from
    several starts, keeping the distinct equilibria whose gap is below 1e-6 minutes.
    """
    if game.K != 1:
        raise ValueError("symmetric_mixed_nash needs a game with one type")
    n, T = game.n, game.T
    comps = np.array(list(compositions(n - 1, T)), dtype=float)  # (C, T) others' loads
    loads = comps[:, None, :] + np.eye(T)[None, :, :]  # (C, t, T): loads if I go to t
    g = game.congestion.cost(loads)  # (C, t, T)
    own = g[:, np.arange(T), np.arange(T)] + game.schedule_cost[0][None, :]  # (C, t) my cost at t
    log_coef = gammaln(n) - gammaln(comps + 1).sum(axis=1)  # multinomial coefficients
    total = (loads * g).sum(axis=2)  # social congestion cost if I go to t, per composition

    def expected(sigma):
        with np.errstate(divide="ignore"):
            logp = log_coef + (comps * np.log(np.maximum(sigma, 1e-300))[None, :]).sum(axis=1)
        p = np.exp(logp - logp.max())
        p /= p.sum()
        return p @ own, p

    rng = np.random.default_rng(seed)
    inits = [np.full(T, 1.0 / T), np.eye(T)[game.tau[0]] * 0.9 + 0.1 / T]
    inits += [rng.dirichlet(np.ones(T)) for _ in range(max(0, starts - len(inits)))]
    found: list[MixedNash] = []
    for s0 in inits:
        sigma = s0.copy()
        for it in range(iters):
            c, _ = expected(sigma)
            avg = sigma @ c
            eta = 0.5 / (1.0 + it / 500.0)
            sigma = sigma * np.exp(-eta * (c - avg))
            sigma = np.maximum(sigma, 0.0)
            sigma /= sigma.sum()
            if it % 50 == 0 and avg - c.min() < 1e-9:
                break
        c, p = expected(sigma)
        gap = float(sigma @ c - c.min())
        if gap > 1e-6:
            continue
        # expected social cost: congestion part (I am one of n identical players) + schedule part
        cong = float(sigma @ (p @ total))  # E[sum_t A_t g_t], averaged over my own draw
        cost = cong + n * float(sigma @ game.schedule_cost[0])
        if not any(np.allclose(sigma, m.sigma, atol=1e-4) for m in found):
            found.append(MixedNash(sigma, cost, gap))
    return sorted(found, key=lambda m: m.cost)


# ----------------------------------------------------------------- exact CE vs Nash
@dataclass
class ExactCE:
    """Exact benchmarks of a small game (all costs are totals over the crowd)."""

    n_schedules: int
    optimum: float
    best_ce: float
    best_cce: float
    best_nash: float  # best pure Nash equilibrium; nan if there is none
    worst_nash: float
    n_nash: int
    mixed_nash: float  # best symmetric mixed Nash equilibrium (one-type games only; nan otherwise)
    ce_support: int  # number of schedules the best CE uses

    @property
    def gap_closed(self) -> float:
        """Share of the (best pure Nash - optimum) gap that the best CE removes (nan if no pure Nash)."""
        if self.n_nash == 0:
            return float("nan")
        gap = self.best_nash - self.optimum
        return float((self.best_nash - self.best_ce) / gap) if gap > 1e-9 else 0.0


def exact_ce(game: SlotGame, tol: float = 1e-9, mixed: bool = True) -> ExactCE:
    """Best CE, best coarse CE, every pure Nash equilibrium and the optimum, by listing every schedule."""
    S = all_schedules(game)
    G, cost = _columns(game, S)
    nash = np.array([game.is_nash(X, tol) for X in S])
    ce, p = _best_ce_lp(G, cost)
    cce, _ = _best_ce_lp(G, cost, _coarse_rows(game, S))
    mn = float("nan")
    if mixed and game.K == 1:
        eqs = symmetric_mixed_nash(game)
        mn = eqs[0].cost if eqs else float("nan")
    return ExactCE(
        n_schedules=len(S), optimum=float(cost.min()), best_ce=ce, best_cce=cce,
        best_nash=float(cost[nash].min()) if nash.any() else float("nan"),
        worst_nash=float(cost[nash].max()) if nash.any() else float("nan"),
        n_nash=int(nash.sum()), mixed_nash=mn, ce_support=int((p > 1e-7).sum()),
    )


# ----------------------------------------------------------------- bad days in a small crowd
@dataclass
class SmallInfo:
    """Expected social cost (total over the crowd) of each way to use the operator's knowledge."""

    tell_nobody: float
    announce: float
    public_hint: float  # best public message, then the best CE of the belief it creates
    private: float  # best Bayes correlated equilibrium
    planner: float  # optimum of each state, always obeyed
    beliefs: np.ndarray
    value: np.ndarray  # V(q): best CE cost when everyone believes "bad with probability q"

    @property
    def privacy_gain(self) -> float:
        """Share of cost that private advice saves over the best public hint."""
        return float(1.0 - self.private / self.public_hint) if self.public_hint > 0 else 0.0

    @property
    def gain_vs_announcements(self) -> float:
        """Share of cost that private advice saves over the better of telling nobody / announcing."""
        return float(1.0 - self.private / min(self.tell_nobody, self.announce))


def small_crowd_information(game: SlotGame, bad_factor: float = 0.7, p_bad: float = 0.3,
                            n_beliefs: int = 41) -> SmallInfo:
    """Exact version of the bad-day comparison (Section 6) for a game small enough to list.

    After any public message the crowd holds a belief q and plays the best CE of the game whose
    congestion cost is the q-mixture of the two states, so V(q) is a best-CE cost and the best public
    hint is the lower convex envelope of V at the prior. Private advice is the best Bayes CE: the
    distribution over schedules may depend on the state, and obedience must hold on average over the
    states in which a given slot is recommended. A public hint followed by CE play is itself a Bayes
    CE, so private <= public hint <= min(tell nobody, announce).
    """
    cm = game.congestion
    if not isinstance(cm, FluidQueue):
        raise ValueError("small_crowd_information needs a fluid-queue game")
    normal = FluidQueue(cm.service_rate, cm.T, cm.slot_minutes)
    bad = FluidQueue(cm.service_rate * bad_factor, cm.T, cm.slot_minutes)
    S = all_schedules(game)
    Gn, cn = _columns(game.with_congestion(normal), S)
    Gb, cb = _columns(game.with_congestion(bad), S)

    q = np.unique(np.r_[np.linspace(0, 1, n_beliefs), p_bad])
    V = np.array([_best_ce_lp((1 - qi) * Gn + qi * Gb, (1 - qi) * cn + qi * cb)[0] for qi in q])
    ip = int(np.flatnonzero(np.isclose(q, p_bad))[0])
    best = V[ip]
    for i in range(ip + 1):  # best chord through the prior: two messages are enough with two states
        for j in range(ip, len(q)):
            if j != i:
                lam = (q[j] - p_bad) / (q[j] - q[i])
                best = min(best, lam * V[i] + (1 - lam) * V[j])

    J = len(S)  # Bayes CE: variables p[w, j], obedience averaged over the states with prior weights
    A_ub = np.hstack([(1 - p_bad) * Gn.T, p_bad * Gb.T])
    obj = np.r_[(1 - p_bad) * cn, p_bad * cb]
    A_eq = np.kron(np.eye(2), np.ones((1, J)))
    res = linprog(obj, A_ub=A_ub, b_ub=np.zeros(A_ub.shape[0]), A_eq=A_eq, b_eq=np.ones(2),
                  bounds=[(0, None)] * (2 * J), method="highs")
    if not res.success:
        raise RuntimeError(f"Bayes-CE LP failed: {res.message}")
    planner = (1 - p_bad) * cn.min() + p_bad * cb.min()
    return SmallInfo(tell_nobody=float(V[ip]), announce=float((1 - p_bad) * V[0] + p_bad * V[-1]),
                     public_hint=float(best), private=float(res.fun), planner=float(planner), beliefs=q, value=V)
