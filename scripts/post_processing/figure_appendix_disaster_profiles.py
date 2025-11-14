"""
Generate appendix figures showing disaster model characteristics.
Simplified and optimized for speed.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.core.disaster import DisasterModel
from scripts.config.params import get_parameters

sns.set(context="paper", style="whitegrid")


def build_model(p_event, p5, p95, n_groups, clusteredness, n_steps=400):
    """Build a single disaster model instance."""
    params = get_parameters(
        {
            "disasterProbability": p_event,
            "numberOfTimeSteps": n_steps,
            "disasterDistributionType": "beta",
            "5th_percentile": p5,
            "95th_percentile": p95,
            "numberOfAgentGroups": n_groups,
            "disasterClusteredness": clusteredness,
        }
    )
    return DisasterModel(params)


def figure_s1(output_dir):
    """Event frequency and impact magnitude distributions."""
    # Use direct binomial for speed
    n_sims, n_steps = 1000, 400
    counts_infreq = np.random.binomial(n_steps + 1, 0.01, n_sims)
    counts_freq = np.random.binomial(n_steps + 1, 0.04, n_sims)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    # Panel A: Event frequency
    bins = np.arange(0, max(counts_infreq.max(), counts_freq.max()) + 2) - 0.5
    axes[0].hist(
        counts_infreq,
        bins=bins,
        alpha=0.7,
        label="Infrequent (p=0.01)",
        color="#1f77b4",
        density=True,
    )
    axes[0].hist(
        counts_freq,
        bins=bins,
        alpha=0.7,
        label="Frequent (p=0.04)",
        color="#ff7f0e",
        density=True,
    )
    axes[0].axvline(
        np.median(counts_infreq),
        ls="--",
        color="#1f77b4",
        label=f"Median={np.median(counts_infreq):.0f}",
    )
    axes[0].axvline(
        np.median(counts_freq),
        ls="--",
        color="#ff7f0e",
        label=f"Median={np.median(counts_freq):.0f}",
    )
    axes[0].set_xlabel("Number of disruption events (T=400)")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Panel A: Event Frequency")
    axes[0].legend()

    # Panel B: Impact PDFs
    model_mod = build_model(0.02, 0.001, 0.5, 8, 0.5)
    model_sev = build_model(0.02, 0.001, 0.65, 8, 0.5)

    x = np.linspace(0.0001, 1.0, 2000)
    axes[1].plot(x, model_mod.distribution.pdf(x), label="Moderate (95th=0.5)")
    axes[1].plot(x, model_sev.distribution.pdf(x), label="Severe (95th=0.65)")
    axes[1].axvline(0.001, ls="--", color="gray", alpha=0.5)
    axes[1].axvline(0.5, ls="--", color="#1f77b4", alpha=0.7)
    axes[1].axvline(0.65, ls="--", color="#ff7f0e", alpha=0.7)
    axes[1].set_xlim(0, 1)
    axes[1].set_xlabel("Impact magnitude")
    axes[1].set_ylabel("PDF")
    axes[1].set_title("Panel B: Impact Distributions")
    axes[1].legend()

    # Panel C: Zoomed
    axes[2].plot(x, model_mod.distribution.pdf(x), label="Moderate")
    axes[2].plot(x, model_sev.distribution.pdf(x), label="Severe")
    axes[2].set_xlim(0, 0.1)
    axes[2].set_xlabel("Impact magnitude")
    axes[2].set_ylabel("PDF")
    axes[2].set_title("Panel C: Zoomed (0-0.1)")
    axes[2].legend()

    fig.suptitle("Figure S1: Disruption Regime Characteristics", y=1.02)
    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "figureS1_disruption_regime_characteristics.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def collect_tau(p_event, p5, p95, n_groups, clusteredness, n_sims=100):
    """Collect tau_g values across simulations."""
    all_tau = []
    for _ in range(n_sims):
        model = build_model(p_event, p5, p95, n_groups, clusteredness)
        mask = model.disaster_impacts > 0
        if np.any(mask):
            all_tau.append(model.turbulenceLevels[:, mask].ravel())
    return np.concatenate(all_tau) if all_tau else np.array([])


def figure_s2(output_dir):
    """Group-level impact distributions."""
    n_groups = 8
    p_event = 0.02  # frequent
    configs = [
        ("Moderate, Uniform", 0.001, 0.5, 0.0),
        ("Moderate, Skewed", 0.001, 0.5, 0.5),
        ("Severe, Uniform", 0.001, 0.65, 0.0),
        ("Severe, Skewed", 0.001, 0.65, 0.5),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    axes = axes.ravel()

    for ax, (title, p5, p95, skew) in zip(axes, configs):
        tau = collect_tau(p_event, p5, p95, n_groups, skew, n_sims=100)
        if tau.size > 0:
            ax.hist(tau, bins=25, color="#1f77b4", alpha=0.8, density=True)
            ax.axvline(1.0, ls="--", color="black", alpha=0.8)
        ax.set_title(f"{title}\n(G={n_groups}, p={p_event})")
        ax.set_xlabel(r"$\tau_g$")
        ax.set_ylabel("Density")

    fig.suptitle(
        f"Figure S2: Group-Level Impact Distributions (G={n_groups}, Frequent)", y=1.02
    )
    fig.tight_layout()

    path = os.path.join(output_dir, "figureS2_group_level_impact_distributions.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def figure_s2b(output_dir):
    """Binned impacts: G × Regime × Skew."""
    G_list = [3, 8, 21]

    # Show 4 key regimes
    regimes = [
        ("Freq-Sev", 0.04, 0.001, 0.65),
        ("Freq-Mod", 0.04, 0.001, 0.5),
        ("Infreq-Sev", 0.01, 0.001, 0.65),
        ("Infreq-Mod", 0.01, 0.001, 0.5),
    ]

    fig, axes = plt.subplots(
        len(regimes), len(G_list), figsize=(14, 11), sharex=True, sharey=True
    )

    for i, (regime_label, p_event, p5, p95) in enumerate(regimes):
        for j, G in enumerate(G_list):
            ax = axes[i, j]

            # Get tau for uniform and skewed
            tau_uniform = collect_tau(p_event, p5, p95, G, 0.0, n_sims=100)
            tau_skewed = collect_tau(p_event, p5, p95, G, 0.5, n_sims=100)

            if tau_uniform.size > 0:
                ax.hist(
                    tau_uniform,
                    bins=25,
                    alpha=0.6,
                    label="Uniform",
                    color="#1f77b4",
                    density=True,
                )
            if tau_skewed.size > 0:
                ax.hist(
                    tau_skewed,
                    bins=25,
                    alpha=0.6,
                    label="Skewed",
                    color="#ff7f0e",
                    density=True,
                )

            ax.axvline(1.0, ls="--", color="black", alpha=0.5, linewidth=1)

            if i == 0:
                ax.set_title(f"G={G}")
            if j == 0:
                ax.set_ylabel(f"{regime_label}\nDensity")
            if i == len(regimes) - 1:
                ax.set_xlabel(r"$\tau_g$")
            if i == 0 and j == 0:
                ax.legend(fontsize=8)

    fig.suptitle("Figure S2b: Post-allocation τ_g by Regime, G, and Skew", y=0.995)
    fig.tight_layout()

    path = os.path.join(output_dir, "figureS2b_binned_impacts.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def figure_s3(output_dir):
    """Skew effect across diversity levels."""
    G_values = [1, 2, 3, 5, 8, 13, 21]
    n_sims = 100

    # Parameters for the regime
    p_event = 0.02  # frequent
    p5, p95 = 0.01, 0.65  # severe impacts

    results = {"Uniform": [], "Skewed": []}

    for sk_name, skew in [("Uniform", 0.0), ("Skewed", 0.5)]:
        for G in G_values:
            fracs = []
            for _ in range(n_sims):
                model = build_model(p_event, p5, p95, G, skew)
                mask = model.disaster_impacts > 0
                if np.any(mask):
                    tau = model.turbulenceLevels[:, mask].ravel()
                    fracs.append(np.mean(tau >= 0.999))
                else:
                    fracs.append(0.0)

            fracs = np.array(fracs)
            med = np.median(fracs)
            lo, hi = np.quantile(fracs, [0.025, 0.975])
            results[sk_name].append((med, lo, hi))

    # Plot
    x = np.arange(len(G_values))
    width = 0.35
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    for idx, sk_name in enumerate(["Uniform", "Skewed"]):
        vals = np.array(results[sk_name])
        meds = vals[:, 0]
        errs_low = meds - vals[:, 1]
        errs_high = vals[:, 2] - meds

        ax.bar(x + (idx - 0.5) * width, meds, width=width, label=sk_name)
        ax.errorbar(
            x + (idx - 0.5) * width,
            meds,
            yerr=[errs_low, errs_high],
            fmt="none",
            ecolor="black",
            capsize=3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([str(g) for g in G_values])
    ax.set_xlabel("Number of groups (G)")
    ax.set_ylabel(r"Median frac($\tau_g$=1.0)")
    ax.set_title(
        f"Figure S3: Skew Effect (Freq-Severe: p={p_event}, 95th={p95})\nn={n_sims} sims, 95% CI"
    )
    ax.set_ylim(0, 1.05)
    ax.legend()

    path = os.path.join(output_dir, "figureS3_skew_effect_across_diversity.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    np.random.seed(42)
    output_dir = os.path.join(PROJECT_ROOT, "results", "fig")

    print("Generating Figure S1...")
    s1 = figure_s1(output_dir)
    print(f"  Saved: {s1}")

    print("Generating Figure S2...")
    s2 = figure_s2(output_dir)
    print(f"  Saved: {s2}")

    print("Generating Figure S2b...")
    s2b = figure_s2b(output_dir)
    print(f"  Saved: {s2b}")

    print("Generating Figure S3...")
    s3 = figure_s3(output_dir)
    print(f"  Saved: {s3}")

    print("\nAll figures saved!")


if __name__ == "__main__":
    main()
