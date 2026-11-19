"""Run every experiment; writes results/figures/*.png, results/RESULTS.md and the website's docs/data.js.

    python experiments.py            # full run (about 10 minutes)
    python experiments.py --quick    # 5 seeds instead of 30 (for a fast check)

General part (five shared facilities: road, network, restaurant, gym, mess):
    E1  small games      Pigou, the two-student dinner game, a traffic light, learning to a CE
    E2  selfish timing   Nash equilibrium vs the planner's optimum (price of anarchy)
    E3  public display   what a shared crowd display does, under three ways of reacting
    E4  CE advice        private recommendations: obedience, best-CE LP, how many must follow, rewards
    E5  uncertainty      who should be told what when the operator knows tonight's state
    E6  learning         students who learn from experience end near a correlated equilibrium
    E8  small crowds     exact: when correlation beats Nash, and when privacy matters, as the crowd grows
    E9  robustness       random arrivals instead of the fluid queue; how bad a "bad day" is and how often
    E10 checks/stats     verification checks and paired permutation tests (Holm) behind the main comparisons
Case study:
    E7  the IITGN mess   synthetic QR-scan logs, calibration, every meal, the dinner day by day
"""
from __future__ import annotations

import argparse
import json
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from crowd import viz
from crowd.correlated import (best_ce, information_design, interpolated_schedules, large_crowd_check, obedience,
                              obedience_incentive, who_gains)
from crowd.domains import DOMAINS, build
from crowd.game import benchmarks, random_arrival_cost
from crowd.smallcrowd import (all_schedules, exact_ce, small_crowd_game, small_crowd_information,
                              symmetric_mixed_nash)
from crowd.game import FluidQueue, SlotGame
from crowd.mess import (MEALS, MESSES, arrival_heatmap, estimate_preferred_slots, estimate_service_rate, generate,
                        make_roster, meal_instance, nowcast_errors, preference_accuracy, roster_truth,
                        scans_per_slot)
from crowd.simulate import (EXP3, Coarse, Forecast, Hedge, Live, Logit, Noisy, NoInfo, Partial, Recommend,
                            RegretMatching, simulate)
from crowd.theory import NormalFormGame, pigou_numeric
from crowd.theory import (critical_responsiveness, dinner_game, hedge, intersection_game, pigou_poa, regret_matching,
                          shared_counter_dynamics)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
FIG = OUT / "figures"
SUMMARY: dict = {}
SITE: dict = {}

REGIMES = {
    "calm": Logit(3.0, 0.3),       # 30% of people reconsider each day
    "reactive": Logit(3.0, 1.0),   # everyone reconsiders each day
    "noisy": Logit(1.0, 0.3),      # low rationality
}
DOMAIN_ORDER = ["road", "network", "restaurant", "gym", "mess"]
NOISE_LEVELS = [3.0, 1.0]  # logit lambda of app users: as rational as the calm crowd, or as the noisy one
MARGINS = [0.0, 1.0, 2.0]  # minutes of reward on top of the obedience reward, so obeying is strictly better
SHORT = {"road": "Road", "network": "Network", "restaurant": "Restaurant", "gym": "Gym", "mess": "Mess"}
POLICY_COLORS = {"no information": viz.MUTED, "public display": viz.NE, "display for 30%": viz.SERIES[3],
                 "CE advice": viz.CE, "optimum as advice": viz.SERIES[4], "CE advice + reward": viz.OPT}


def ci(x) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    return float(x.mean()), float(1.96 * x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0


def save(fig, name: str) -> None:
    fig.savefig(FIG / f"{name}.png")
    viz.plt.close(fig)


def plain_log(ax, axis: str, ticks) -> None:
    """Log axis with ordinary numbers (2, 5, 10) instead of 10^x labels."""
    from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator

    a = ax.xaxis if axis == "x" else ax.yaxis
    (ax.set_xscale if axis == "x" else ax.set_yscale)("log")
    a.set_major_locator(FixedLocator(ticks))
    a.set_major_formatter(lambda v, _: f"{v:g}")
    a.set_minor_locator(NullLocator())
    a.set_minor_formatter(NullFormatter())


def seq_cmap():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("seq_blue", ["#f4f8fd"] + viz.SEQ_BLUE)


# ------------------------------------------------------------------ shared set-up (computed once)
@lru_cache(maxsize=None)
def domain(key: str):
    """Instance, equilibrium/optimum benchmarks, and what a tempted person would do, for one domain."""
    inst = build(DOMAINS[key])
    b = benchmarks(inst.game)
    ob_opt = obedience(inst.game, b.optimum.X)
    ob_ne = obedience(inst.game, b.best_nash.X)
    g = inst.game
    alt_opt = g.deviation_costs(b.optimum.X).argmin(axis=2)
    alt_ne = g.deviation_costs(b.best_nash.X).argmin(axis=2)
    return inst, b, ob_ne, ob_opt, alt_ne, alt_opt


@lru_cache(maxsize=None)
def jaiswal(day: str = "normal"):
    inst = meal_instance("Jaiswal", "dinner", day)
    b = benchmarks(inst.game)
    g = inst.game
    return (inst, b, obedience(g, b.best_nash.X), obedience(g, b.optimum.X),
            g.deviation_costs(b.best_nash.X).argmin(axis=2), g.deviation_costs(b.optimum.X).argmin(axis=2))


def core_policies(setup) -> dict:
    """The six information policies compared in every domain."""
    inst, b, ob_ne, ob_opt, alt_ne, alt_opt = setup
    t = inst.type_of
    tol = max(0.1, b.epsilon) + 1e-9  # disobey only if it saves more than 6 s (or than the schedule's own accuracy)
    return {
        "no information": NoInfo(),
        "public display": Live(),
        "display for 30%": Partial(Live(), 0.3),
        "CE advice": Recommend(target=b.best_nash.X, type_of=t, rational=True, gains=ob_ne.gain_by_cell,
                               best_alt=alt_ne, tolerance=tol, name="CE advice"),
        "optimum as advice": Recommend(target=b.optimum.X, type_of=t, rational=True, gains=ob_opt.gain_by_cell,
                                       best_alt=alt_opt, name="optimum as advice"),
        "CE advice + reward": Recommend(target=b.optimum.X, type_of=t, rational=True, gains=ob_opt.gain_by_cell,
                                        best_alt=alt_opt, bonus=ob_opt.gain_by_cell, name="CE advice + reward"),
    }


def run(inst, policy, regime: str, seed: int, days: int = 150, burn: int = 50):
    g = inst.game
    return simulate(g.congestion, inst.tau, inst.beta, policy, REGIMES[regime], days=days, burn_in=burn, seed=seed)


# =============================================================== E1 small games
def e1_small_games() -> None:
    plt = viz.plt
    g = dinner_game()
    mixed = [m for m in g.mixed_nash_2p() if 0 < m[0][0] < 1][0]
    counter = float(np.mean([g.total_cost(s) for s in shared_counter_dynamics(g, (1, 1), 20)]))
    mixed_cost = float(g.expected_costs(g.product_distribution(list(mixed))).sum())
    ce = g.correlated_equilibrium("fair")
    ig = intersection_game()
    best = ig.correlated_equilibrium("best")
    nash_payoffs = [-ig.expected_costs(ig.product_distribution([x, y])) for x, y in ig.mixed_nash_2p()]
    rm = regret_matching(ig, rounds=6000, seed=0, log_every=25)
    hd = hedge(ig, rounds=6000, seed=0, log_every=25)
    SUMMARY["e1"] = {
        "dinner": {"shared counter": counter, "no information (mixed Nash)": mixed_cost,
                   "private coin (CE)": ce.total_cost, "optimum": g.social_optimum()[1], "mixed_peak_prob": float(mixed[0][0])},
        "intersection": {"best_ce_each": float(-best.expected_costs[0]), "best_symmetric_nash": float(max(min(u) for u in nash_payoffs)),
                         "support": best.support()},
        "pigou": {str(p): pigou_poa(p) for p in (1, 2, 4, 8)},
        "learning": {"regret_matching_ce_gap": float(rm.ce_violation[-1]), "hedge_cce_gap": float(hd.cce_violation[-1])},
        "delay": {str(d): critical_responsiveness(d) for d in range(5)},
    }
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.5))
    vals = [counter, mixed_cost, ce.total_cost]
    ax[0].bar(range(3), vals, 0.6, color=[viz.NE, viz.MUTED, viz.CE])
    ax[0].axhline(g.social_optimum()[1], color=viz.OPT, ls="--", lw=1.4)
    ax[0].text(2.45, g.social_optimum()[1] + 0.06, "optimum", color=viz.OPT, ha="right", fontsize=9)
    for i, v in enumerate(vals):
        ax[0].text(i, v + 0.06, f"{v:.1f}", ha="center", fontsize=10, color=viz.TEXT_2)
    ax[0].set_xticks(range(3), ["shared counter", "no information\n(mixed Nash)", "private coin\n(correlated eq.)"])
    ax[0].set(title="Two students, Peak or Late: total cost", ylim=(0, 4.2))
    outcomes = ["both stop", "one goes", "crash"]
    ce_p = [best.p[0], best.p[1] + best.p[2], best.p[3]]
    mix = [m for m in ig.mixed_nash_2p() if 0 < m[0][0] < 1][0]
    ne_p = ig.product_distribution(list(mix))
    ne_p = [ne_p[0], ne_p[1] + ne_p[2], ne_p[3]]
    x = np.arange(3)
    ax[1].bar(x - 0.2, ce_p, 0.38, color=viz.CE, label=f"correlated eq. ({-best.expected_costs[0]:.2f} each)")
    ax[1].bar(x + 0.2, ne_p, 0.38, color=viz.NE, label=f"mixed Nash ({14 / 3:.2f} each)")
    ax[1].set_xticks(x, outcomes)
    ax[1].set(title="A traffic light: how often each outcome happens", ylabel="probability", ylim=(0, 0.75))
    ax[1].legend(fontsize=8.5)
    ax[2].plot(rm.rounds_logged, rm.ce_violation, color=viz.CE, label="regret matching (CE gap)")
    ax[2].plot(hd.rounds_logged, hd.cce_violation, color=viz.SERIES[3], label="Hedge (coarse-CE gap)")
    ax[2].set(title="Players who learn find a correlated equilibrium", xlabel="rounds played", yscale="log",
              ylabel="largest gain from deviating")
    ax[2].legend(fontsize=8.5)
    fig.tight_layout()
    save(fig, "fig1_small_games")


# =============================================================== E2 selfish timing
def e2_selfish() -> None:
    plt = viz.plt
    rows = []
    for key in DOMAIN_ORDER:
        inst, b, *_ = domain(key)
        g, d = inst.game, DOMAINS[key]
        ne, op = g.breakdown(b.nash.X), g.breakdown(b.optimum.X)
        rows.append({"domain": key, "title": d.title, "people": g.n, "unit": d.unit, "slot_min": d.slot_minutes,
                     "service_rate": d.service_rate, "nash": ne["per_person"], "nash_wait": ne["wait_per_person"],
                     "opt": op["per_person"], "opt_wait": op["wait_per_person"], "poa": b.poa,
                     "eps_sec": 60 * b.epsilon})
    SUMMARY["e2"] = {"domains": rows}
    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 3.7), gridspec_kw={"width_ratios": [1, 1.25]})
    ax = axes[0]
    x = np.arange(len(df))
    ax.bar(x - 0.2, df.nash, 0.38, color=viz.NE, label="selfish (Nash)")
    ax.bar(x + 0.2, df.opt, 0.38, color=viz.OPT, label="planner (optimum)")
    for xi, r in zip(x, rows):
        ax.text(xi, r["nash"] * 1.12, f"{r['poa']:.1f}x", ha="center", fontsize=9, color=viz.TEXT_2)
    ax.set_xticks(x, [SHORT[k] for k in df.domain])
    plain_log(ax, "y", [0.3, 1, 3, 10, 30])
    ax.set(title="Minutes lost per person, and the price of anarchy", ylabel="minutes per person (log)", ylim=(0.2, 40))
    ax.legend(loc="upper right", fontsize=8.5)
    inst, b, *_ = domain("road")
    g = inst.game
    xs = np.arange(g.T)
    ax = axes[1]
    ax.bar(xs - 0.2, g.loads(b.nash.X), 0.38, color=viz.NE, label="selfish (Nash)")
    ax.bar(xs + 0.2, g.loads(b.optimum.X), 0.38, color=viz.OPT, label="planner (optimum)")
    ax.axhline(g.congestion.capacity, color=viz.MUTED, ls="--", lw=1.2)
    ax.text(g.T - 0.5, g.congestion.capacity + 6, "bridge capacity per slot", ha="right", color=viz.TEXT_2, fontsize=9)
    ax.set_xticks(xs[::2], g.slot_labels[::2])
    ax.set(title="Morning commute: cars arriving per 15 minutes", ylabel="cars")
    ax.legend(loc="upper left", fontsize=8.5)
    fig.tight_layout()
    save(fig, "fig2_selfish_vs_planner")


# =============================================================== E3 public display
def e3_display(seeds: int) -> None:
    plt = viz.plt
    rows = []
    for key in DOMAIN_ORDER:
        setup = domain(key)
        inst, b = setup[0], setup[1]
        nash = b.best_nash.social_cost / inst.game.n
        for regime in REGIMES:
            for name, pol in core_policies(setup).items():
                ms = [run(inst, pol, regime, s).metrics for s in range(seeds)]
                c, h = ci([m["mean_cost"] for m in ms])
                rows.append({"domain": key, "regime": regime, "policy": name, "cost": c, "ci": h, "vs_nash": c / nash,
                             "per_seed": [m["mean_cost"] for m in ms],
                             "swing": float(np.mean([m["swing"] for m in ms])),
                             "regret": float(np.mean([m["swap_regret"] for m in ms])),
                             "p90": float(np.mean([m["p90_wait"] for m in ms]))})
    SUMMARY["e3"] = rows
    df = pd.DataFrame(rows)
    pols = list(POLICY_COLORS)
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.9), sharey=True)
    w = 0.13
    for ax, regime in zip(axes, REGIMES):
        sub = df[df.regime == regime]
        for j, pol in enumerate(pols):
            vals = [sub[(sub.domain == k) & (sub.policy == pol)].vs_nash.iloc[0] for k in DOMAIN_ORDER]
            ax.bar(np.arange(5) + (j - 2.5) * w, vals, w * 0.92, color=POLICY_COLORS[pol], label=pol)
        ax.axhline(1, color=viz.TEXT_2, lw=1)
        ax.set_xticks(range(5), [SHORT[k] for k in DOMAIN_ORDER])
        ax.set_title({"calm": "Calm crowd (30% reconsider daily)", "reactive": "Reactive crowd (everyone reconsiders)",
                      "noisy": "Noisy crowd (low rationality)"}[regime])
        plain_log(ax, "y", [0.1, 0.3, 1, 3, 10])
    axes[0].set_ylabel("cost relative to Nash (log)")
    axes[0].legend(fontsize=7.5, loc="upper left", ncol=2)
    fig.tight_layout()
    save(fig, "fig3_public_display")


# =============================================================== E4 correlated advice at scale
def e4_ce(seeds: int) -> None:
    plt = viz.plt
    table, comp = [], []
    shares = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]
    for key in DOMAIN_ORDER:
        setup = domain(key)
        inst, b, ob_ne, ob_opt, alt_ne, alt_opt = setup
        g = inst.game
        n = g.n
        sched, labels = interpolated_schedules(g, b.best_nash.X, b.optimum.X, steps=20)
        sched += [s.X for s in b.all_nash]
        atomic = best_ce(g, sched, labels, slack=b.epsilon)
        lc = large_crowd_check(g)
        inc = obedience_incentive(g, b.optimum.X, b.best_nash.social_cost)
        wg = who_gains(g, b.best_nash.X, b.optimum.X)
        table.append({"domain": key, "nash": b.best_nash.social_cost / n, "opt": b.optimum.social_cost / n,
                      "ce_max_gain_sec": 60 * ob_ne.max_gain, "opt_tempted": ob_opt.share_tempted,
                      "opt_max_gain_min": ob_opt.max_gain, "best_ce_atomic": atomic.expected_cost / n,
                      "eps_sec": 60 * b.epsilon, "large_nash": lc.nash, "large_best_ce": lc.best_ce,
                      "large_gap_sec": lc.gap_sec, "reward_pp": inc.per_person, "rewarded": inc.share_rewarded,
                      "reward_per_rewarded": inc.per_person / max(inc.share_rewarded, 1e-9),
                      "saving_pp": inc.welfare_gain / n, "worse_off_opt": wg.worse_off("optimum"),
                      "worse_off_reward": wg.worse_off("with_reward"), "max_loss_opt_sec": 60 * wg.max_loss("optimum")})
        nash = b.best_nash.social_cost / n
        for sh in shares:
            pol = Recommend(target=b.best_nash.X, type_of=inst.type_of, compliance=sh, rational=True,
                            gains=ob_ne.gain_by_cell, best_alt=alt_ne, fallback=Live(),
                            tolerance=max(0.1, b.epsilon) + 1e-9)
            c, h = ci([run(inst, pol, "reactive", s).metrics["mean_cost"] for s in range(max(3, seeds // 3))])
            comp.append({"domain": key, "share": sh, "cost": c, "ci": h, "vs_nash": c / nash})
    # app users as noisy as everyone else (logit on the costs the app can compute), with a safety margin
    noisy = []
    for key in DOMAIN_ORDER:
        inst, b, ob_ne, ob_opt, alt_ne, alt_opt = domain(key)
        g = inst.game
        nash = b.best_nash.social_cost / g.n
        D_ne, D_opt = g.deviation_costs(b.best_nash.X), g.deviation_costs(b.optimum.X)
        for lam in NOISE_LEVELS:
            for m in MARGINS:
                pols = {"CE advice": Recommend(target=b.best_nash.X, type_of=inst.type_of, noise=lam, dev_costs=D_ne,
                                               bonus=np.full_like(ob_ne.gain_by_cell, m)),
                        "CE advice + reward": Recommend(target=b.optimum.X, type_of=inst.type_of, noise=lam,
                                                        dev_costs=D_opt, bonus=ob_opt.gain_by_cell + m)}
                for name, pol in pols.items():
                    per = [run(inst, pol, "calm", s, days=60, burn=10).metrics["mean_cost"]
                           for s in range(max(3, seeds // 3))]
                    c, h = ci(per)
                    X = b.best_nash.X if name == "CE advice" else b.optimum.X
                    budget = float((X * pol.bonus).sum() / g.n)
                    noisy.append({"domain": key, "lam": lam, "margin": m, "policy": name, "cost": c, "ci": h,
                                  "vs_nash": c / nash, "budget_pp": budget, "per_seed": per})
    SUMMARY["e4"] = {"table": table, "compliance": comp, "noisy": noisy}
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 3.7))
    ax = axes[0]
    cdf = pd.DataFrame(comp)
    for key, col in zip(DOMAIN_ORDER, [viz.SERIES[0], viz.SERIES[1], viz.SERIES[2], viz.SERIES[3], viz.SERIES[6]]):
        sub = cdf[cdf.domain == key]
        ax.plot(sub.share, sub.vs_nash, marker="o", ms=4.5, color=col, label=SHORT[key])
    ax.axhline(1, color=viz.TEXT_2, lw=1)
    plain_log(ax, "y", [1, 2, 5, 10, 20])
    ax.set(title="Reactive crowd: share of people using private advice", xlabel="share using the app (others see the display)",
           ylabel="cost relative to Nash (log)")
    ax.legend(fontsize=8.5)
    ax = axes[1]
    t = pd.DataFrame(table)
    x = np.arange(len(t))
    ax.bar(x - 0.2, t.reward_pp, 0.38, color=viz.SERIES[3], label="reward paid (average per person)")
    ax.bar(x + 0.2, t.saving_pp, 0.38, color=viz.OPT, label="time saved vs Nash (per person)")
    for xi, r in zip(x, table):
        ax.text(xi - 0.2, r["reward_pp"] * 1.15, f"{r['rewarded']:.0%}\npaid", ha="center", fontsize=7.5, color=viz.TEXT_2)
    ax.set_xticks(x, [SHORT[k] for k in t.domain])
    plain_log(ax, "y", [0.1, 0.3, 1, 3, 10])
    ax.set(title="Price of obedience: reward needed vs time saved", ylabel="minutes per person (log)", ylim=(0.08, 30))
    ax.legend(fontsize=8.5, loc="upper right")
    fig.tight_layout()
    save(fig, "fig4_ce_advice")


# =============================================================== E5 uncertainty / information design
def e5_information() -> None:
    plt = viz.plt
    out = {}
    for key in DOMAIN_ORDER:
        inst, *_ = domain(key)
        r = information_design(inst.game)
        out[key] = {"expected": r.expected, "per_state": r.per_state, "hint": r.hint, "support": r.private_support,
                    "beliefs": r.beliefs, "value": r.value, "envelope": r.envelope, "max_gap_sec": r.max_gap_sec,
                    "tol_sec": r.tol_sec, "menu": r.menu_size, "bad_state": DOMAINS[key].bad_state}
    SUMMARY["e5"] = out
    pols = ["tell nobody", "announce the state", "best public hint", "private advice"]
    cols = [viz.MUTED, viz.NE, viz.SERIES[3], viz.CE]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 3.8), gridspec_kw={"width_ratios": [1.3, 1]})
    ax = axes[0]
    w = 0.19
    for j, (pol, c) in enumerate(zip(pols, cols)):
        vals = [out[k]["expected"][pol] / out[k]["expected"]["tell nobody"] for k in DOMAIN_ORDER]
        ax.bar(np.arange(5) + (j - 1.5) * w, vals, w * 0.92, color=c, label=pol)
    ax.axhline(1, color=viz.TEXT_2, lw=1)
    ax.set_xticks(range(5), [SHORT[k] for k in DOMAIN_ORDER])
    ax.set(title="Expected cost relative to telling nobody", ylabel="relative cost", ylim=(0, 1.42))
    ax.legend(fontsize=8, ncol=2, loc="upper left")
    ax = axes[1]
    r = out["road"]
    q, V, E = np.array(r["beliefs"]), np.array(r["value"]), np.array(r["envelope"])
    p0 = 0.3
    ax.plot(q, V, color=viz.TEXT_2, lw=1.8, label="cost if everyone believes q")
    ax.plot(q, E, color=viz.SERIES[3], lw=2, ls="--", label="best public hint (convex envelope)")
    ax.plot([0, 1], [V[0], V[-1]], color=viz.NE, lw=1.2, ls=":", label="announce the state")
    ax.scatter([p0], [r["expected"]["tell nobody"]], color=viz.MUTED, s=40, zorder=3)
    ax.scatter([p0], [r["expected"]["announce the state"]], color=viz.NE, s=40, zorder=3)
    ax.scatter([p0], [r["expected"]["best public hint"]], color=viz.SERIES[3], s=40, zorder=3)
    ax.axvline(p0, color=viz.GRID, lw=1)
    ax.set(title="Road: cost as a function of the crowd's belief", xlabel="belief q that today is a bad day",
           ylabel="expected minutes per car")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    save(fig, "fig5_uncertainty")


# =============================================================== E6 learning
def e6_learning(seeds: int) -> None:
    rows = []
    learners = {"Hedge (sees every slot's wait)": lambda: Hedge(0.05),
                "Regret matching (sees every slot's wait)": lambda: RegretMatching(),
                "EXP3 (sees only its own wait)": lambda: EXP3(0.05)}
    for key in DOMAIN_ORDER:
        inst, b, *_ = domain(key)
        g = inst.game
        nash = b.best_nash.social_cost / g.n
        for name, make in learners.items():
            ms = [simulate(g.congestion, inst.tau, inst.beta, Live(), make(), days=300, burn_in=100, seed=s).metrics
                  for s in range(max(3, seeds // 10))]
            rows.append({"domain": key, "learner": name, "vs_nash": float(np.mean([m["mean_cost"] for m in ms])) / nash,
                         "swap_regret": float(np.mean([m["swap_regret"] for m in ms])),
                         "swing": float(np.mean([m["swing"] for m in ms]))})
    SUMMARY["e6"] = rows


# =============================================================== E7 the IITGN mess (case study)
def e7_mess(seeds: int) -> None:
    plt = viz.plt
    # --- data and calibration
    scans, truth = generate(weeks=8, seed=0)
    rt = roster_truth(0).set_index("student_id")
    calib = []
    for meal in MEALS:
        for menu in ("normal", "special"):
            true_mu = MEALS[meal].service_rate * (0.8 if menu == "special" else 1.0)
            est = estimate_service_rate(scans, "Jaiswal", meal, menu)
            w = truth[(truth.meal == meal) & (truth.menu == menu) & (truth.mess == "Jaiswal")].wait_min
            calib.append({"meal": meal, "menu": menu, "true_mu": true_mu, "estimated_mu": round(est, 2),
                          "queue_forms": bool(w.max() > 1)})
    prefs = {meal: preference_accuracy(estimate_preferred_slots(scans, meal), rt[f"tau_{meal}"]) for meal in MEALS}
    now = []
    for meal in MEALS:
        for menu in ("normal", "special"):
            errs = [nowcast_errors(scans_per_slot(scans, m, meal, menu)) for m in MESSES]
            now.append({"meal": meal, "menu": menu, **{k: float(np.mean([e[k] for e in errs])) for k in errs[0]}})
    # --- every mess, meal and day
    roster = make_roster(0)
    meals = []
    for mess in MESSES:
        for meal in MEALS:
            for day in ("normal", "special"):
                inst = meal_instance(mess, meal, day, roster=roster)
                g = inst.game
                b = benchmarks(g)
                ne, op = g.breakdown(b.nash.X), g.breakdown(b.optimum.X)
                rn = random_arrival_cost(g, b.nash.X, reps=50) / g.n
                ro = random_arrival_cost(g, b.optimum.X, reps=50) / g.n
                meals.append({"mess": mess, "meal": meal, "day": day, "diners": g.n, "nash": ne["per_person"],
                              "opt": op["per_person"], "lost": ne["per_person"] - op["per_person"],
                              "lost_random": rn - ro,
                              "poa": b.poa if ne["per_person"] >= 0.1 else None, "eps_sec": 60 * b.epsilon})
    mdf = pd.DataFrame(meals)
    hours = float((mdf.lost * mdf.diners)[mdf.day == "normal"].sum() / 60)
    hours_random = float((mdf.lost_random * mdf.diners)[mdf.day == "normal"].sum() / 60)
    agg = mdf.groupby(["meal", "day"])[["nash", "opt", "lost"]].mean().reset_index()
    agg = agg.sort_values(["meal", "day"], key=lambda c: c.map({m: i for i, m in enumerate(MEALS)}) if c.name == "meal" else c)
    agg = agg.reset_index(drop=True)  # breakfast, lunch, dinner; normal before special
    # --- Jaiswal dinner: all display policies (the brief's list) x three crowds
    setup = jaiswal("normal")
    inst, b, ob_ne, ob_opt, alt_ne, alt_opt = setup
    g = inst.game
    nash = b.best_nash.social_cost / g.n
    extra = {"display 3 dinners old": Live(delay=3), "display + private noise": Noisy(3.0),
             "display in 3 levels": Coarse(3), "moving-average forecast": Forecast(0.3)}
    pols = {**core_policies(setup), **extra}
    jrows = []
    for regime in REGIMES:
        for name, pol in pols.items():
            ms = [run(inst, pol, regime, s).metrics for s in range(seeds)]
            c, h = ci([m["mean_cost"] for m in ms])
            jrows.append({"regime": regime, "policy": name, "cost": c, "ci": h, "swing": float(np.mean([m["swing"] for m in ms])),
                          "p90": float(np.mean([m["p90_wait"] for m in ms])),
                          "regret": float(np.mean([m["swap_regret"] for m in ms]))})
    # --- sweeps (reactive crowd) and delay x reactivity
    sweeps = []
    for name, items in {"noise": [(v, Noisy(v)) for v in (0, 1, 2, 3, 5, 8, 12)],
                        "share": [(v, Partial(Live(), v)) for v in (0.0, 0.1, 0.3, 0.5, 0.7, 1.0)],
                        "levels": [(v, Coarse(v)) for v in (2, 3, 5, 10, 30)]}.items():
        for val, pol in items:
            c, h = ci([run(inst, pol, "reactive", s).metrics["mean_cost"] for s in range(seeds)])
            sweeps.append({"sweep": name, "value": val, "cost": c, "ci": h})
    revise = [0.1, 0.2, 0.3, 0.5, 0.75, 1.0]
    delays = [1, 2, 3, 4, 5]
    swing = np.zeros((len(revise), len(delays)))
    for i, r in enumerate(revise):
        for j, d in enumerate(delays):
            swing[i, j] = np.mean([simulate(g.congestion, inst.tau, inst.beta, Live(delay=d), Logit(3.0, r), days=150,
                                            burn_in=50, seed=s).metrics["swing"] for s in range(max(3, seeds // 3))])
    info = information_design(g)
    wg = who_gains(g, b.best_nash.X, b.optimum.X)
    peak = (g.tau >= 3) & (g.tau <= 6)
    groups = {"flexible students": g.beta == 1.5, "average": g.beta == 3.0, "rigid students": g.beta == 6.0,
              "prefer 20:00-20:30": peak, "prefer other times": ~peak}
    gains = [{"group": nm, "nash": wg.group_mean("nash", m), "optimum": wg.group_mean("optimum", m),
              "with_reward": wg.group_mean("with_reward", m)} for nm, m in groups.items()]
    SUMMARY["e7"] = {"rows": int(len(scans)), "students": int(scans.student_id.nunique()), "calibration": calib,
                     "preferences": prefs, "nowcast": now, "meals": meals, "by_meal": agg.to_dict("records"),
                     "hours_lost_per_day": hours, "hours_lost_random": hours_random, "jaiswal": {"nash": nash, "opt": b.optimum.social_cost / g.n,
                                                             "policies": jrows}, "sweeps": sweeps,
                     "swing": {"revise": revise, "delays": delays, "values": swing.round(4).tolist()},
                     "information": {"expected": info.expected, "per_state": info.per_state, "hint": info.hint},
                     "who_gains": gains, "worse_off": {"optimum": wg.worse_off("optimum"),
                                                       "with_reward": wg.worse_off("with_reward")},
                     "max_loss_sec": 60 * wg.max_loss("optimum"), "budget_pp": wg.budget / g.n}

    # --- figure: data and every meal
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 3.5), gridspec_kw={"width_ratios": [1.15, 1]})
    H = arrival_heatmap(scans, "Jaiswal", "dinner").iloc[:, :12]
    SITE["heatmap"] = {"days": list(H.index), "times": list(H.columns), "values": H.values.round(1).tolist()}
    im = axes[0].imshow(H.values, aspect="auto", cmap=seq_cmap(), vmin=0)
    axes[0].set_xticks(range(0, 12, 2), H.columns[::2])
    axes[0].set_yticks(range(len(H.index)), H.index)
    axes[0].grid(False)
    axes[0].set_title("Jaiswal dinner: QR scans per 10 minutes (8 weeks)")
    fig.colorbar(im, ax=axes[0], shrink=0.85, label="scans")
    ax = axes[1]
    labels = [f"{r['meal']}\n{r['day']}" for r in agg.to_dict("records")]
    x = np.arange(len(labels))
    ax.bar(x - 0.2, agg.nash, 0.38, color=viz.NE, label="selfish (Nash)")
    ax.bar(x + 0.2, agg.opt, 0.38, color=viz.OPT, label="planner")
    for xi, v in zip(x, agg.nash):
        ax.text(xi - 0.2, v + 0.15, f"{v:.1f}", ha="center", fontsize=8, color=viz.TEXT_2)
    ax.set_xticks(x, labels, fontsize=8.5)
    ax.set(title="Cost per student per meal (average of 4 messes)", ylabel="minutes")
    ax.legend(fontsize=8.5)
    fig.tight_layout()
    save(fig, "fig6_mess_data")

    # --- figure: the dinner day by day (reactive crowd)
    show = {"no information": pols["no information"], "public display": pols["public display"],
            "CE advice": pols["CE advice"]}
    cap = g.congestion.capacity
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 4.6), gridspec_kw={"height_ratios": [2.2, 1]})
    from matplotlib.colors import TwoSlopeNorm
    norm = TwoSlopeNorm(vmin=0, vcenter=1, vmax=3)
    for j, (name, pol) in enumerate(show.items()):
        res = run(inst, pol, "reactive", 0, days=90, burn=50)
        A = res.loads[50:90] / cap
        im = axes[0, j].imshow(A, aspect="auto", cmap="RdBu_r", norm=norm, interpolation="nearest")
        axes[0, j].set_xticks(range(0, 12, 3), g.slot_labels[::3])
        axes[0, j].set_yticks([0, 10, 20, 30, 39], [51, 61, 71, 81, 90])
        axes[0, j].grid(False)
        axes[0, j].set_title(f"{name}: arrivals / capacity", fontsize=10.5)
        axes[1, j].plot(np.arange(51, 91), np.nanmean(res.cost[50:90], axis=1), color=POLICY_COLORS[name], lw=1.8)
        axes[1, j].set(ylim=(0, 16), xlabel="dinner")
        axes[1, j].text(52, 13.5, f"swing {res.metrics['swing']:.0%}", fontsize=9, color=viz.TEXT_2)
    axes[0, 0].set_ylabel("dinner")
    axes[1, 0].set_ylabel("cost (min)")
    fig.colorbar(im, ax=axes[0, :], shrink=0.9, label="arrivals / capacity (red = overload)")
    save(fig, "fig7_mess_dinners")

    # --- figure: how to show information at the mess
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 3.4), gridspec_kw={"width_ratios": [1.25, 1]})
    sdf = pd.DataFrame(sweeps)
    ax = axes[0]
    sub = sdf[sdf.sweep == "share"]
    ax.errorbar(sub.value, sub.cost, yerr=sub.ci, color=viz.CE, marker="o", ms=5, capsize=2, label="share who see the display")
    ax.axhline(nash, color=viz.NE, ls="--", lw=1.2)
    ax.text(0.01, nash * 1.08, "Nash", color=viz.NE, fontsize=9)
    ax.set(title="Reactive crowd: who should see the public display?", xlabel="share of students who see it",
           ylabel="minutes per student")
    ax = axes[1]
    im = ax.imshow(swing, origin="lower", aspect="auto", cmap=seq_cmap())
    ax.set_xticks(range(len(delays)), delays)
    ax.set_yticks(range(len(revise)), revise)
    ax.grid(False)
    for i in range(len(revise)):
        for j in range(len(delays)):
            ax.text(j, i, f"{swing[i, j]:.0%}", ha="center", va="center", fontsize=8,
                    color="white" if swing[i, j] > swing.max() * 0.55 else viz.TEXT)
    ax.set(xlabel="display delay (dinners)", ylabel="share reacting each day", title="Crowd swing: share switching slot daily")
    save(fig, "fig8_mess_display")


# =============================================================== E8 small crowds, solved exactly
SMALL_SIZES = list(range(2, 61))
INFO_SIZES = [6, 8, 10, 12, 16, 20, 30, 40, 60, 80, 120]
SEVERITIES = [0.7, 0.5]  # bad day: the bottleneck runs at 70% (as in E5) or at 50% of its rate
VARIANT_SIZES = [4, 6, 8, 10, 12, 16, 20, 30, 40]


def _two_types(n: int) -> SlotGame:
    """Half flexible (beta 1), half rigid (beta 4), all preferring the middle slot."""
    h = n // 2
    return SlotGame(FluidQueue(n / (0.8 * 3 * 10.0), T=3, slot_minutes=10.0), tau=[1, 1], beta=[1.0, 4.0],
                    counts=[h, n - h])


def _two_groups(n: int) -> SlotGame:
    """Two thirds prefer the middle slot, one third the late slot (beta 2)."""
    h = (2 * n) // 3
    return SlotGame(FluidQueue(n / (0.8 * 3 * 10.0), T=3, slot_minutes=10.0), tau=[1, 2], beta=[2.0, 2.0],
                    counts=[h, n - h])


SMALL_VARIANTS = {
    "base: 1 type, 3 slots, load 0.8, beta 2": lambda n: small_crowd_game(n),
    "lighter load (0.6)": lambda n: small_crowd_game(n, load=0.6),
    "full load (1.0)": lambda n: small_crowd_game(n, load=1.0),
    "more flexible (beta 1)": lambda n: small_crowd_game(n, beta=1.0),
    "more rigid (beta 4)": lambda n: small_crowd_game(n, beta=4.0),
    "4 slots": lambda n: small_crowd_game(n, T=4),
    "2 types: flexible and rigid": _two_types,
    "2 groups: middle and late": _two_groups,
}


def e8_small_crowds() -> None:
    plt = viz.plt
    rows = []
    for n in SMALL_SIZES:
        r = exact_ce(small_crowd_game(n))
        rows.append({"n": n, "schedules": r.n_schedules, "optimum": r.optimum / n, "best_cce": r.best_cce / n,
                     "best_ce": r.best_ce / n, "best_pure_nash": r.best_nash / n, "mixed_nash": r.mixed_nash / n,
                     "pure_nash_count": r.n_nash, "gap_closed": r.gap_closed, "ce_support": r.ce_support})
    df = pd.DataFrame(rows)
    has = df[df.pure_nash_count > 0]
    info = []
    for slow in SEVERITIES:
        for n in INFO_SIZES:
            r = small_crowd_information(small_crowd_game(n), bad_factor=slow)
            info.append({"slow": slow, "n": n, "tell_nobody": r.tell_nobody / n, "announce": r.announce / n,
                         "public_hint": r.public_hint / n, "private": r.private / n, "planner": r.planner / n,
                         "privacy_gain": r.privacy_gain})
    limit = {}
    for slow in SEVERITIES:  # the same game in the large-crowd limit (Section 6's method)
        e = information_design(small_crowd_game(120), bad_factor=slow).expected
        limit[str(slow)] = {"public_hint": e["best public hint"], "private": e["private advice"],
                            "privacy_gain": 1 - e["private advice"] / e["best public hint"]}
    idf = pd.DataFrame(info)
    variants = []
    for name, make in SMALL_VARIANTS.items():
        vr = []
        for n in VARIANT_SIZES:
            g = make(n)
            r = exact_ce(g, mixed=(g.K == 1))
            vr.append({"n": n, "pure": r.n_nash > 0, "beats_pure": r.n_nash > 0 and r.best_ce < r.best_nash - 1e-6,
                       "saving_pure": 1 - r.best_ce / r.best_nash if r.n_nash else float("nan"),
                       "saving_mixed": 1 - r.best_ce / r.mixed_nash if g.K == 1 else float("nan")})
        v = pd.DataFrame(vr)
        withp = v[v.pure]
        variants.append({"crowd": name, "sizes": len(v), "with_pure_nash": int(v.pure.sum()),
                         "ce_beats_pure": int(withp.beats_pure.sum()),
                         "max_saving_pure": float(withp.saving_pure.max()) if len(withp) else float("nan"),
                         "min_saving_mixed": float(v.saving_mixed.min()), "max_saving_mixed": float(v.saving_mixed.max())})
    best = has.loc[has.gap_closed.idxmax()]
    SUMMARY["e8"] = {
        "table": df[df.n.isin([2, 4, 6, 8, 10, 12, 16, 20, 30, 40, 60])].round(3).to_dict("records"),
        "strictly_better": int((has.best_pure_nash - has.best_ce > 1e-6).sum()), "with_pure_nash": int(len(has)),
        "no_pure_nash": [int(n) for n in df[df.pure_nash_count == 0].n],
        "best_n": int(best.n), "best_gap_closed": float(best.gap_closed), "best_row": best.to_dict(),
        "gap_closed_51_60": float(df[df.n > 50].gap_closed.mean()),
        "mixed_over_ce_60": float(df[df.n == 60].mixed_nash.iloc[0] / df[df.n == 60].best_ce.iloc[0]),
        "info": idf.round(4).to_dict("records"), "limit": limit, "variants": variants,
    }

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 3.7))
    ax = axes[0]
    ax.plot(df.n, df.mixed_nash, color=viz.MUTED, marker="o", ms=2.5, lw=1.4, label="mixed Nash (no coordination)")
    ax.plot(has.n, has.best_pure_nash, color=viz.NE, marker="o", ms=2.5, lw=1.4, label="best pure Nash")
    ax.plot(df.n, df.best_ce, color=viz.CE, marker="o", ms=2.5, lw=1.8, label="best correlated equilibrium")
    ax.plot(df.n, df.optimum, color=viz.OPT, marker="o", ms=2.5, lw=1.4, label="planner")
    ax.set(title="Identical people, 3 slots: exact equilibria by crowd size", xlabel="crowd size n (same load at every size)",
           ylabel="minutes per person", ylim=(1.0, 4.0))
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    ax = axes[1]
    for slow, c in zip(SEVERITIES, [viz.CE, viz.SERIES[6]]):
        sub = idf[idf.slow == slow]
        ax.plot(sub.n, 100 * sub.privacy_gain, color=c, marker="o", ms=4, label=f"bad day at {slow:.0%} capacity")
        ax.axhline(100 * limit[str(slow)]["privacy_gain"], color=c, ls="--", lw=1)
    ax.plot([], [], color=viz.TEXT_2, ls="--", lw=1, label="same game, large-crowd limit")
    plain_log(ax, "x", [6, 10, 20, 40, 80, 120])
    ax.set(title="Bad days: what private advice saves over the best public hint", xlabel="crowd size n (log)",
           ylabel="saving (%)", ylim=(0, None))
    ax.legend(fontsize=8.5)
    fig.tight_layout()
    save(fig, "fig9_small_crowds")


# =============================================================== E9 robustness
SWEEP_SEVERITY = [0.5, 0.7, 0.85]
SWEEP_PROB = [0.1, 0.3, 0.5]


def e9_robustness() -> None:
    plt = viz.plt
    arrivals = []
    for key in DOMAIN_ORDER:
        inst, b, *_ = domain(key)
        g = inst.game
        n = g.n
        row = {"domain": key}
        for lab, X in (("worst_nash", b.nash.X), ("best_nash", b.best_nash.X), ("opt", b.optimum.X)):
            row[f"{lab}_fluid"] = g.social_cost(X) / n
            row[f"{lab}_random"] = random_arrival_cost(g, X, reps=200) / n
            row[f"{lab}_random_exp"] = random_arrival_cost(g, X, reps=200, service="exponential") / n
        for m in ("fluid", "random", "random_exp"):
            row[f"poa_{m}"] = row[f"worst_nash_{m}"] / row[f"opt_{m}"]
            row[f"lost_{m}"] = row[f"worst_nash_{m}"] - row[f"opt_{m}"]
        arrivals.append(row)
    sweep = []
    for key in DOMAIN_ORDER:
        g = domain(key)[0].game
        for slow in SWEEP_SEVERITY:
            if slow == 0.7 and "e5" in SUMMARY:
                e = SUMMARY["e5"][key]["expected"]
            else:
                e = information_design(g, bad_factor=slow).expected
            sweep.append({"domain": key, "slow": slow, "p_bad": 0.3, **e})
    for pb in SWEEP_PROB:
        if pb != 0.3:
            e = information_design(domain("road")[0].game, p_bad=pb).expected
            sweep.append({"domain": "road", "slow": 0.7, "p_bad": pb, **e})
    sdf = pd.DataFrame(sweep)
    sdf["announce_vs_nobody"] = sdf["announce the state"] / sdf["tell nobody"]
    sdf["privacy_gain"] = 1 - sdf["private advice"] / sdf["best public hint"]
    adf = pd.DataFrame(arrivals)
    SUMMARY["e9"] = {"arrivals": adf.round(3).to_dict("records"), "sweep": sdf.round(4).to_dict("records")}

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 3.6))
    ax = axes[0]
    x = np.arange(len(adf))
    for j, (m, lab, c) in enumerate((("fluid", "fluid queue (model)", viz.NE),
                                     ("random", "random arrivals", viz.SERIES[3]),
                                     ("random_exp", "random arrivals and service", viz.MUTED))):
        ax.bar(x + (j - 1) * 0.26, adf[f"poa_{m}"], 0.24, color=c, label=lab)
    ax.axhline(1, color=viz.TEXT_2, lw=1)
    ax.set_xticks(x, [SHORT[k] for k in adf.domain])
    ax.set(title="Price of anarchy when arrivals are random", ylabel="worst Nash / planner")
    ax.legend(fontsize=8.5)
    ax = axes[1]
    for key, c in zip(DOMAIN_ORDER, [viz.SERIES[0], viz.SERIES[1], viz.SERIES[2], viz.SERIES[3], viz.SERIES[6]]):
        sub = sdf[(sdf.domain == key) & (sdf.p_bad == 0.3)].sort_values("slow")
        ax.plot(100 * sub.slow, sub.announce_vs_nobody, color=c, marker="o", ms=4.5, label=SHORT[key])
    ax.axhline(1, color=viz.TEXT_2, lw=1)
    ax.axhspan(1, 1.5, color=viz.NE, alpha=0.06, lw=0, label="announcing backfires")
    ax.set(title="Announcing a bad day, by how bad it is (probability 0.3)", xlabel="capacity on a bad day (%)",
           ylabel="announce / tell nobody")
    ax.legend(fontsize=8.5, ncol=2)
    fig.tight_layout()
    save(fig, "fig10_robustness")


# =============================================================== checks and paired statistics
def sign_flip_p(d, R: int = 20000, seed: int = 0) -> float:
    """Two-sided paired sign-flip permutation test of mean(d) = 0."""
    d = np.asarray(d, dtype=float)
    if np.allclose(d, 0):
        return 1.0
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(R, len(d)))
    obs = abs(d.mean())
    return float((1 + (np.abs((signs * d).mean(axis=1)) >= obs - 1e-12).sum()) / (R + 1))


def holm(pvals) -> list:
    """Holm step-down adjusted p-values."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    run_max = 0.0
    for rank, i in enumerate(order):
        run_max = max(run_max, (len(p) - rank) * p[i])
        adj[i] = min(1.0, run_max)
    return adj.tolist()


def e10_checks_and_stats() -> None:
    """Verification checks (Table 'checks') and paired tests behind the main comparisons."""
    checks = []

    def add(name, must, result, ok):
        checks.append({"check": name, "must": must, "result": result, "pass": bool(ok)})

    add("Pigou, linear costs", "price of anarchy 4/3 by formula and numerically",
        f"{pigou_poa(1):.6f} and {pigou_numeric(1)['poa']:.6f}",
        abs(pigou_poa(1) - 4 / 3) < 1e-9 and abs(pigou_numeric(1)["poa"] - 4 / 3) < 1e-6)
    g6 = small_crowd_game(6)
    n6, T = g6.n, g6.T
    costs = np.zeros((n6,) + (T,) * n6)
    for prof in np.ndindex(*(T,) * n6):
        c = g6.costs(np.bincount(prof, minlength=T)[None, :])[0]
        for i in range(n6):
            costs[(i, *prof)] = c[prof[i]]
    nf = NormalFormGame(costs)
    r6 = exact_ce(g6, mixed=False)
    full = nf.correlated_equilibrium("best").total_cost
    add("Exact small-crowd CE (6 people)", "schedule LP = LP over all $3^6$ profiles",
        f"difference {abs(r6.best_ce - full):.1e} min", abs(r6.best_ce - full) < 1e-6)
    g2 = small_crowd_game(2)
    T2 = g2.T
    c2 = np.zeros((2, T2, T2))
    for a in range(T2):
        for b_ in range(T2):
            c = g2.costs(np.bincount([a, b_], minlength=T2)[None, :])[0]
            c2[0, a, b_], c2[1, a, b_] = c[a], c[b_]
    nf2 = NormalFormGame(c2)
    sym = [x for x, y in nf2.mixed_nash_2p() if np.allclose(x, y)]
    ours = symmetric_mixed_nash(g2)
    diff = float(np.abs(ours[0].sigma - sym[0]).max()) if sym and ours else float("nan")
    add("Symmetric mixed Nash (2 people)", "same as support enumeration", f"largest difference {diff:.1e}", diff < 1e-5)
    e4 = SUMMARY["e4"]["table"]
    worst = max(r["ce_max_gain_sec"] - r["eps_sec"] for r in e4)
    add("Proposition 1 (5 facilities)", "gain from disobeying CE advice $\\le$ equilibrium accuracy $\\varepsilon$",
        f"largest excess {max(worst, 0):.1f} s", worst <= 1e-6)
    rest = 0.0
    for key in DOMAIN_ORDER:
        inst, b, _, ob_opt, *_ = domain(key)
        G = inst.game.gains(b.optimum.X)
        rest = max(rest, float(np.max(G - ob_opt.gain_by_cell[:, :, None])))
    add("Proposition 3 (5 facilities)", "after the reward nobody gains by moving", f"largest gain {max(rest, 0):.1e} min",
        rest <= 1e-9)
    add("Proposition 2, large crowd", "best CE over the menu = Nash", 
        f"largest gap {max(r['large_gap_sec'] for r in e4):.2f} s", max(r["large_gap_sec"] for r in e4) < 1.0)
    m = FluidQueue(1.0, T=2, slot_minutes=10.0)
    from crowd.game import random_arrival_wait
    rel = abs(random_arrival_wait(m, np.array([60, 0]), reps=400)[0] / m.wait(np.array([60.0, 0.0]))[0] - 1)
    add("Random-arrival queue", "matches the fluid queue under heavy load", f"relative difference {rel:.1%}", rel < 0.05)
    viol = 0.0
    for r in SUMMARY["e8"]["info"]:
        viol = max(viol, r["private"] - r["public_hint"], r["public_hint"] - min(r["tell_nobody"], r["announce"]))
    for r in SUMMARY["e9"]["sweep"]:
        viol = max(viol, r["private advice"] - r["best public hint"],
                   r["best public hint"] - min(r["tell nobody"], r["announce the state"]))
    add("Bad days (all runs)", "private $\\le$ best public hint $\\le$ min(silence, announcing)",
        f"largest violation {max(viol, 0):.1e} min", viol <= 1e-6)
    one = small_crowd_information(small_crowd_game(10), bad_factor=1.0)
    d1 = abs(one.private - one.tell_nobody) / 10
    add("Bayes CE with one state", "= the best CE", f"difference {d1:.1e} min", d1 < 1e-6)

    # paired tests (seeds are shared across policies, so differences are paired)
    e3 = pd.DataFrame(SUMMARY["e3"])
    nz = pd.DataFrame(SUMMARY["e4"]["noisy"])
    fams = {}

    def per(df, **kw):
        q = df
        for k, v in kw.items():
            q = q[q[k] == v]
        return np.array(q.per_seed.iloc[0])

    rows = []
    for key in DOMAIN_ORDER:
        r3 = e3[(e3.domain == key) & (e3.regime == "reactive")]
        rows.append(("display vs CE advice (reactive)", key,
                     per(r3, policy="public display") - per(r3, policy="CE advice")))
    for key in DOMAIN_ORDER:
        r3 = e3[(e3.domain == key) & (e3.regime == "reactive")]
        rows.append(("display for 30% vs display (reactive)", key,
                     per(r3, policy="display for 30%") - per(r3, policy="public display")))
    for key in DOMAIN_ORDER:
        sub = nz[(nz.domain == key) & (nz.lam == 3.0) & (nz.policy == "CE advice + reward")]
        rows.append(("reward + 2-min margin vs exact reward (noisy users)", key,
                     per(sub, margin=2.0) - per(sub, margin=0.0)))
    tests = []
    for fam, key, d in rows:
        tests.append({"family": fam, "domain": key, "mean_diff": float(d.mean()), "p": sign_flip_p(d), "n": len(d)})
    tdf = pd.DataFrame(tests)
    tdf["p_holm"] = np.nan
    for fam, idx in tdf.groupby("family").groups.items():
        tdf.loc[idx, "p_holm"] = holm(tdf.loc[idx, "p"])
    SUMMARY["e10"] = {"checks": checks, "tests": tdf.to_dict("records")}


def _tex(x: str) -> str:
    return x.replace("%", "\\%")


def write_report_files(minutes: float) -> None:
    """report/generated/: one LaTeX macro per number quoted in the new sections, and generated tables."""
    S = SUMMARY
    gen = ROOT / "report" / "generated"
    gen.mkdir(parents=True, exist_ok=True)
    arr = {r["domain"]: r for r in S["e9"]["arrivals"]}
    sweep = S["e9"]["sweep"]
    noisy = S["e4"]["noisy"]
    e4t = {r["domain"]: r for r in S["e4"]["table"]}
    e8, e7 = S["e8"], S["e7"]
    NAME = {"road": "Road", "network": "Network", "restaurant": "Food court", "gym": "Gym", "mess": "Mess"}

    def rng(vals, d=1):
        lo, hi = min(vals), max(vals)
        return f"{lo:.{d}f}" if f"{lo:.{d}f}" == f"{hi:.{d}f}" else f"{lo:.{d}f}--{hi:.{d}f}"

    def nz(policy, lam, m, key=None):
        return [r["vs_nash"] for r in noisy if r["policy"] == policy and r["lam"] == lam and r["margin"] == m
                and (key is None or r["domain"] == key)]

    sev = [r for r in sweep if r["slow"] == 0.5 and r["p_bad"] == 0.3]
    clear = [r for r in sev if r["announce_vs_nobody"] > 1.05]
    marginal = [r for r in sev if 1 < r["announce_vs_nobody"] <= 1.05]
    roadp = sorted([r for r in sweep if r["domain"] == "road" and r["slow"] == 0.7], key=lambda r: r["p_bad"])
    mess50 = [r for r in sev if r["domain"] == "mess"][0]
    tests = pd.DataFrame(S["e10"]["tests"])
    var = pd.DataFrame(e8["variants"])
    one_type = var[var.min_saving_mixed.notna()]
    macros = {
        "PoaRandom": rng([arr[k]["poa_random"] for k in DOMAIN_ORDER]),
        "PoaRandomExp": rng([arr[k]["poa_random_exp"] for k in DOMAIN_ORDER]),
        "LostRoad": f"{arr['road']['lost_random']:.1f}", "LostMess": f"{arr['mess']['lost_random']:.1f}",
        "HoursFluid": f"{e7['hours_lost_per_day']:.0f}", "HoursRandom": f"{e7['hours_lost_random']:.0f}",
        "EightBetter": str(e8["strictly_better"]), "EightWith": str(e8["with_pure_nash"]),
        "EightFade": f"{100 * e8['gap_closed_51_60']:.0f}\\%", "EightMixed": f"{e8['mixed_over_ce_60']:.2f}",
        "EightBestGap": f"{100 * e8['best_gap_closed']:.0f}\\%",
        "CERange": rng(nz("CE advice", 3.0, 0.0) + nz("CE advice", 1.0, 0.0)),
        "RwThreeMzero": rng(nz("CE advice + reward", 3.0, 0.0)),
        "RwThreeMtwo": rng(nz("CE advice + reward", 3.0, 2.0), 2),
        "RwOneMtwo": rng(nz("CE advice + reward", 1.0, 2.0), 2),
        "OptRange": rng([e4t[k]["opt"] / e4t[k]["nash"] for k in DOMAIN_ORDER], 2),
        "BackfireClear": str(len(clear)), "BackfireMarginal": str(len(marginal)),
        "BackfireList": ", ".join(f"{NAME[r['domain']].lower()} (+{100 * (r['announce_vs_nobody'] - 1):.0f}\\%)"
                                  for r in clear),
        "BackfireMarginalList": ", ".join(f"{NAME[r['domain']].lower()} (+{100 * (r['announce_vs_nobody'] - 1):.0f}\\%)"
                                          for r in marginal),
        "RoadP": ", ".join(f"probability {r['p_bad']:g}: +{100 * (r['announce_vs_nobody'] - 1):.0f}\\%" for r in roadp),
        "PrivMax": f"{100 * max(r['privacy_gain'] for r in sweep):.1f}\\%",
        "MessFiftyNobody": f"{mess50['tell nobody']:.1f}", "MessFiftyAnn": f"{mess50['announce the state']:.1f}",
        "MessFiftyHint": f"{mess50['best public hint']:.1f}",
        "RunMinutes": f"{round(minutes / 5) * 5:.0f}",
        "NoisySeeds": str(len(noisy[0]["per_seed"])),
        "TestsSig": str(int((tests.p_holm < 0.05).sum())), "TestsAll": str(len(tests)),
        "TestsMaxP": f"{tests.p_holm.max():.3f}",
        "VarCount": str(len(var)), "VarSizes": str(len(VARIANT_SIZES)),
        "VarMixedMin": f"{100 * one_type.min_saving_mixed.min():.0f}\\%",
        "VarMixedMax": f"{100 * one_type.max_saving_mixed.max():.0f}\\%",
        "ChecksPass": str(sum(c["pass"] for c in S["e10"]["checks"])), "ChecksAll": str(len(S["e10"]["checks"])),
        "VarBeat": str(int((var.ce_beats_pure > 0).sum())),
        "PrivSmallSeventy": rng([100 * r["privacy_gain"] for r in e8["info"] if r["slow"] == 0.7], 0) + "\\%",
        "PrivSmallFifty": rng([100 * r["privacy_gain"] for r in e8["info"] if r["slow"] == 0.5], 0) + "\\%",
        "PrivSmallMax": f"{100 * max(r['privacy_gain'] for r in e8['info']):.0f}\\%",
        "PrivLimitSeventy": f"{100 * e8['limit']['0.7']['privacy_gain']:.2f}\\%",
        "PrivLimitFifty": f"{100 * e8['limit']['0.5']['privacy_gain']:.1f}\\%",
    }
    (gen / "numbers.tex").write_text("% generated by `python experiments.py` -- do not edit by hand\n" +
                                     "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in macros.items()))
    rows = []
    for k in DOMAIN_ORDER:
        vals = [nz("CE advice", 3.0, 0.0, k)[0], nz("CE advice", 1.0, 0.0, k)[0]]
        vals += [nz("CE advice + reward", 3.0, m, k)[0] for m in MARGINS]
        vals += [nz("CE advice + reward", 1.0, m, k)[0] for m in MARGINS]
        vals += [e4t[k]["opt"] / e4t[k]["nash"]]
        rows.append(NAME[k] + " & " + " & ".join(f"{v:.2f}" for v in vals) + r"\\")
    (gen / "tab_noisy.tex").write_text("\n".join(rows) + "\n")
    rows = []
    for r in e8["variants"]:
        mixed = "--" if r["min_saving_mixed"] != r["min_saving_mixed"] else \
            f"{100 * r['min_saving_mixed']:.0f}--{100 * r['max_saving_mixed']:.0f}\\%"
        best = "--" if r["max_saving_pure"] != r["max_saving_pure"] else f"{100 * r['max_saving_pure']:.0f}\\%"
        rows.append(f"{r['crowd']} & {r['with_pure_nash']} of {r['sizes']} & {r['ce_beats_pure']} of "
                    f"{r['with_pure_nash']} & {best} & {mixed}\\\\")
    (gen / "tab_variants.tex").write_text("\n".join(rows) + "\n")
    rows = [f"{c['check']} & {c['must']} & {_tex(c['result'])} & {'pass' if c['pass'] else 'FAIL'}\\\\"
            for c in S["e10"]["checks"]]
    (gen / "tab_checks.tex").write_text("\n".join(rows) + "\n")


# =============================================================== website data
REPO = "https://github.com/YashG1836/crowd-congestion"
APP = "https://crowd-congestion.streamlit.app/"
SITE_URL = "https://yashg1836.github.io/crowd-congestion/"


def export_site() -> None:
    """Write docs/data.js: every number and series the website shows, produced by this code."""
    playback = {}
    for key in DOMAIN_ORDER:
        setup = domain(key)
        inst = setup[0]
        g = inst.game
        playback[key] = {"labels": g.slot_labels, "capacity": g.congestion.capacity, "unit": DOMAINS[key].unit,
                         "policies": {}}
        for name in ("no information", "public display", "display for 30%", "CE advice", "CE advice + reward"):
            res = run(inst, core_policies(setup)[name], "reactive", 0, days=120, burn=40)
            playback[key]["policies"][name] = {"loads": res.loads.astype(int).tolist(),
                                               "cost": np.nanmean(res.cost, axis=1).round(3).tolist(),
                                               "swing": res.metrics["swing"], "mean": res.metrics["mean_cost"]}
    jsetup = jaiswal("normal")
    inst, b, ob_ne, ob_opt, *_ = jsetup
    g = inst.game
    playback["jaiswal"] = {"labels": g.slot_labels, "capacity": g.congestion.capacity, "unit": "students",
                           "policies": {}}
    for name in ("no information", "public display", "display for 30%", "CE advice", "CE advice + reward"):
        res = run(inst, core_policies(jsetup)[name], "reactive", 0, days=120, burn=40)
        playback["jaiswal"]["policies"][name] = {"loads": res.loads.astype(int).tolist(),
                                                 "cost": np.nanmean(res.cost, axis=1).round(3).tolist(),
                                                 "swing": res.metrics["swing"], "mean": res.metrics["mean_cost"]}

    def phone(X, k, t, bonus=0.0):
        D = g.deviation_costs(X)[k, t].copy()  # this student's cost in every slot if everyone else obeys
        D[t] -= bonus
        return {"tau": int(g.tau[k]), "rec": int(t), "costs": D.round(3).tolist(), "beta": float(g.beta[k])}

    k_opt, t_opt = map(int, np.unravel_index(np.argmax(ob_opt.gain_by_cell), ob_opt.gain_by_cell.shape))
    k_ne = int(np.argmax(g.counts * (g.tau == 4)))
    t_ne = int(np.argmax(b.best_nash.X[k_ne]))
    phones = {"nash": phone(b.best_nash.X, k_ne, t_ne), "opt": phone(b.optimum.X, k_opt, t_opt),
              "reward": phone(b.optimum.X, k_opt, t_opt, float(ob_opt.gain_by_cell[k_opt, t_opt]))}
    mess_sched = {"labels": g.slot_labels, "capacity": g.congestion.capacity,
                  "nash": g.loads(b.nash.X).astype(int).tolist(), "opt": g.loads(b.optimum.X).astype(int).tolist(),
                  "wait_nash": g.congestion.wait(g.loads(b.nash.X).astype(float)).round(2).tolist(),
                  "wait_opt": g.congestion.wait(g.loads(b.optimum.X).astype(float)).round(2).tolist()}
    dom_sched = {}
    for key in DOMAIN_ORDER:
        di, db, *_ = domain(key)
        G = di.game
        dom_sched[key] = {"labels": G.slot_labels, "capacity": G.congestion.capacity,
                          "nash": G.loads(db.nash.X).astype(int).tolist(), "opt": G.loads(db.optimum.X).astype(int).tolist(),
                          "story": DOMAINS[key].story, "title": DOMAINS[key].title, "unit": DOMAINS[key].unit,
                          "bad_state": DOMAINS[key].bad_state}
    data = {"links": {"repo": REPO, "app": APP, "site": SITE_URL, "report": REPO + "/blob/main/report/main.pdf"},
            "summary": SUMMARY, "playback": playback, "phones": phones, "mess_schedule": mess_sched,
            "domains": dom_sched, "order": DOMAIN_ORDER, **SITE}
    (ROOT / "docs").mkdir(exist_ok=True)
    text = json.dumps(data, default=_jsonable, separators=(",", ":"))
    (ROOT / "docs" / "data.js").write_text("// generated by `python experiments.py` -- do not edit by hand\n"
                                           "window.DATA = " + text + ";\n")


def _jsonable(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    return str(o)


# =============================================================== RESULTS.md
def md(df: pd.DataFrame, digits: int = 3) -> str:
    return df.round(digits).to_markdown(index=False)


def write_markdown() -> None:
    s = SUMMARY
    L = ["# Results (generated by `python experiments.py`)", "",
         "Costs are minutes per person per visit (waiting + inconvenience of going away from one's preferred time). "
         "Synthetic data. Intervals are 95% confidence half-widths over random seeds.", ""]
    e1 = s["e1"]
    L += ["## E1 Small games", "", md(pd.DataFrame([e1["dinner"]])), "",
          f"Traffic light: best correlated equilibrium gives each driver {e1['intersection']['best_ce_each']:.2f}; the best "
          f"symmetric Nash outcome gives {e1['intersection']['best_symmetric_nash']:.2f}. Learning: regret matching ends "
          f"{e1['learning']['regret_matching_ce_gap']:.4f} from the CE set, Hedge {e1['learning']['hedge_cce_gap']:.4f} "
          "from the coarse-CE set.", ""]
    L += ["## E2 Selfish timing vs the planner", "", md(pd.DataFrame(s["e2"]["domains"]).drop(columns=["title"])), ""]
    e3 = pd.DataFrame(s["e3"])
    piv = e3.pivot_table(index=["regime", "policy"], columns="domain", values="vs_nash").reindex(columns=DOMAIN_ORDER)
    L += ["## E3 Public display: cost relative to Nash", "", piv.round(2).to_markdown(), "",
          "Crowd swing (share of people switching slot each day), reactive crowd:", "",
          e3[e3.regime == "reactive"].pivot_table(index="policy", columns="domain", values="swing")
          .reindex(columns=DOMAIN_ORDER).round(3).to_markdown(), ""]
    L += ["## E4 Correlated advice at scale", "", md(pd.DataFrame(s["e4"]["table"])), "",
          "Compliance (reactive crowd, non-users see the display), cost relative to Nash:", "",
          pd.DataFrame(s["e4"]["compliance"]).pivot_table(index="share", columns="domain", values="vs_nash")
          .reindex(columns=DOMAIN_ORDER).round(2).to_markdown(), ""]
    nz = pd.DataFrame(s["e4"]["noisy"])
    L += ["App users as noisy as everyone else (logit lambda), with a reward margin of m minutes; cost relative to Nash:",
          "", nz.pivot_table(index=["policy", "lam", "margin"], columns="domain", values="vs_nash")
          .reindex(columns=DOMAIN_ORDER).round(2).to_markdown(), "", "Reward budget, minutes per person:", "",
          nz.pivot_table(index=["policy", "margin"], columns="domain", values="budget_pp")
          .reindex(columns=DOMAIN_ORDER).round(2).to_markdown(), ""]
    L += ["## E5 Who should be told what (bad day: capacity 70%, probability 0.3; large-crowd limit)", ""]
    rows = []
    for k, v in s["e5"].items():
        rows.append({"domain": k, **{p: round(c, 3) for p, c in v["expected"].items()},
                     "P(busy|bad)": round(v["hint"]["P(busy | bad)"], 2),
                     "P(busy|normal)": round(v["hint"]["P(busy | normal)"], 2),
                     "tolerance (s)": round(v["tol_sec"], 2)})
    L += [md(pd.DataFrame(rows)), "", "Per-state costs (road):", "",
          pd.DataFrame(s["e5"]["road"]["per_state"]).round(2).to_markdown(), ""]
    L += ["## E6 Learning", "", md(pd.DataFrame(s["e6"])), ""]
    e7 = s["e7"]
    L += ["## E7 The IITGN mess", "",
          f"{e7['rows']:,} QR scans for {e7['students']} students over 8 weeks; {e7['hours_lost_per_day']:.0f} "
          "student-hours lost to selfish timing on a normal day (4 messes, 3 meals).", "",
          md(pd.DataFrame(e7["calibration"])), "", pd.DataFrame(e7["preferences"]).T.round(3).to_markdown(), "",
          md(pd.DataFrame(e7["by_meal"])), "", "Jaiswal dinner, every display policy:", "",
          pd.DataFrame(e7["jaiswal"]["policies"]).pivot_table(index="policy", columns="regime", values="cost")
          .round(2).to_markdown(), "", md(pd.DataFrame(e7["sweeps"])), "",
          "Information design at Jaiswal dinner:", "", md(pd.DataFrame([e7["information"]["expected"]])), "",
          "Who gains:", "", md(pd.DataFrame(e7["who_gains"])), "",
          f"Worse off than Nash: optimum without reward {e7['worse_off']['optimum']} students (by at most "
          f"{e7['max_loss_sec']:.1f} s), with reward {e7['worse_off']['with_reward']}. Reward budget "
          f"{e7['budget_pp']:.2f} min per student.", "", "Nowcasting MAE (scans per slot):", "", md(pd.DataFrame(e7["nowcast"])), ""]
    L[-1:-1] = [f"With random arrival times inside each slot the same schedules lose "
                f"{e7['hours_lost_random']:.0f} student-hours a day instead of {e7['hours_lost_per_day']:.0f}.", ""]
    e8 = s["e8"]
    br = e8["best_row"]
    L += ["## E8 Small crowds, solved exactly", "",
          "Identical people, 3 slots, everyone prefers the middle one, load 0.8. Every schedule is listed, so all values"
          " are exact (minutes per person).", "",
          f"- The best CE is strictly cheaper than every pure Nash equilibrium in {e8['strictly_better']} of the"
          f" {e8['with_pure_nash']} sizes from 2 to 60 that have one; no pure Nash exists for n = {e8['no_pure_nash']}.",
          f"- Largest advantage at n = {e8['best_n']}: pure Nash {br['best_pure_nash']:.2f}, best CE {br['best_ce']:.2f},"
          f" planner {br['optimum']:.2f} ({e8['best_gap_closed']:.0%} of the gap closed). For n = 51-60 the best CE"
          f" closes {e8['gap_closed_51_60']:.1%} of the gap.",
          f"- Against people who cannot coordinate at all (symmetric mixed Nash) the best CE is still"
          f" {e8['mixed_over_ce_60']:.2f}x cheaper at n = 60.", "", md(pd.DataFrame(e8["table"])), "",
          "Bad days in a small crowd (probability 0.3), minutes per person:", "", md(pd.DataFrame(e8["info"])), "",
          "Same game in the large-crowd limit: " + "; ".join(
              f"bad day at {float(k):.0%}: private advice saves {v['privacy_gain']:.2%} over the best public hint"
              for k, v in e8["limit"].items()) + ".", ""]
    e9 = s["e9"]
    L += ["## E9 Robustness", "", "Random arrival times inside each slot (fluid schedules re-evaluated by simulation):",
          "", md(pd.DataFrame(e9["arrivals"])), "", "Bad-day sweep (large-crowd limit, expected minutes per person):", "",
          md(pd.DataFrame(e9["sweep"])), ""]
    L += ["Small crowds of other shapes (crowd sizes " + ", ".join(map(str, VARIANT_SIZES)) + "):", "",
          md(pd.DataFrame(e8["variants"])), ""]
    e10 = s["e10"]
    L += ["## E10 Verification checks and paired tests", "", md(pd.DataFrame(e10["checks"])), "",
          "Paired sign-flip permutation tests over seeds, Holm-adjusted within each family:", "",
          md(pd.DataFrame(e10["tests"]), 4), ""]
    (OUT / "RESULTS.md").write_text("\n".join(L))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="5 seeds instead of 30")
    ap.add_argument("--report-only", action="store_true",
                    help="rebuild report/generated/ from docs/data.js without re-running the experiments")
    ap.add_argument("--minutes", type=float, default=None, help="with --report-only: run time to quote")
    args = ap.parse_args()
    if args.report_only:
        text = (ROOT / "docs" / "data.js").read_text()
        SUMMARY.update(json.loads(text[text.index("=") + 1:].rstrip().rstrip(";"))["summary"])
        write_report_files(args.minutes if args.minutes is not None else float(SUMMARY.get("runtime_min", 0.0)))
        print("wrote report/generated/")
        return
    seeds = 5 if args.quick else 30
    FIG.mkdir(parents=True, exist_ok=True)
    viz.style()
    steps = [("E1 small games", e1_small_games), ("E2 selfish timing", e2_selfish),
             ("E3 public display", lambda: e3_display(seeds)), ("E4 CE advice", lambda: e4_ce(seeds)),
             ("E5 uncertainty", e5_information), ("E6 learning", lambda: e6_learning(seeds)),
             ("E7 the mess", lambda: e7_mess(seeds)), ("E8 small crowds", e8_small_crowds),
             ("E9 robustness", e9_robustness), ("E10 checks and statistics", e10_checks_and_stats),
             ("website data", export_site)]
    start = time.time()
    SUMMARY["runtime_min"] = 0.0
    for name, fn in steps:
        if name == "website data":
            SUMMARY["runtime_min"] = (time.time() - start) / 60
        t0 = time.time()
        fn()
        print(f"{name:20s} done in {time.time() - t0:6.1f}s", flush=True)
    write_markdown()
    write_report_files((time.time() - start) / 60)
    print(f"wrote {OUT} and docs/data.js")


if __name__ == "__main__":
    main()
