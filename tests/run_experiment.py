import argparse
from pathlib import Path

import jax
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t
from tqdm import tqdm
import matplotlib as mpl
mpl.rcParams["text.usetex"] = False

from mapc_cmab.agents.hierarchical_dqn import HierarchicalMapcDQNAgent
from mapc_cmab.agents.mapc_cmab_agent_factory import MapcDQNAgentFactory
from mapc_cmab.envs.scenario_impl import small_office_scenario
from mapc_cmab.loggers.action_reward_logger import Logger 
from mapc_cmab.plots.throughput_analysis.throughput_ci import analyze_and_plot_throughputs

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run sequential Hierarchical DQN MAPC experiments."
    )

    parser.add_argument("--d-ap", type=float, default=10.0,
                        help="AP-to-AP distance used by the scenario.")
    parser.add_argument("--d-sta", type=float, default=2.0,
                        help="Station distance used by the scenario.")
    parser.add_argument("--n-runs", type=int, default=10,
                        help="Number of independent runs.")
    parser.add_argument("--n-steps", type=int, default=5000,
                        help="Number of simulation steps per run.")
    parser.add_argument("--n-links", type=int, default=3,
                        help="Number of available links.")
    parser.add_argument("--n-tx-power-levels", type=int, default=4,
                        help="Number of transmission-power levels.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base random seed.")
    parser.add_argument("--window-size", type=int, default=100,
                        help="Number of steps per throughput averaging window.")
    parser.add_argument("--step-duration", type=float, default=0.005,
                        help="Duration of one simulation step in seconds.")
    parser.add_argument("--confidence", type=float, default=0.99,
                        help="Confidence level for the Student-t interval.")
    parser.add_argument("--output-dir", type=str, default="./results",
                        help="Directory for output files.")
    parser.add_argument("--filename", type=str, default=None,
                        help="Base name for output files. Auto-generated if omitted.")
    parser.add_argument("--show", action="store_true",
                        help="Display the throughput plot.")

    return parser.parse_args()


def create_agent_factory(scenario, args):
    return MapcDQNAgentFactory(
        associations=scenario.associations,
        agent_params_lvl1=None,
        agent_params_lvl2=None,
        agent_params_lvl3=None,
        agent_params_lvl4=None,
        n_tx_power_levels=args.n_tx_power_levels,
        n_links=args.n_links,
        seed=args.seed,
    )
def run_single_experiment(agent_factory, scenario, run_number, n_steps, key):
    logger = Logger(run_number=run_number, exp_name=scenario.str_repr)
    agent = agent_factory.create_hierarchical_DQN_cmapc_agent(logger=logger)

    throughputs = np.zeros(n_steps, dtype=np.float32)
    sta_throughputs_history = np.zeros((n_steps, 16), dtype=np.float32)
    fairness_history = np.zeros(n_steps, dtype=np.float32)

    previous_throughput = 0.0

    for step in tqdm(
        range(1, n_steps),
        desc=f"run_number: {run_number}",
        leave=True
    ):
        key, step_key = jax.random.split(key)

        tx_config = agent.sample(reward=previous_throughput)

        data_rate, reward, band_rates, tx_matrices = scenario(
            step_key,
            tx_config,
            return_per_sta=True
        )

        data_rate = float(data_rate)

        # --------------------------------
        # Build 16-STA throughput vector
        # --------------------------------
        sta_throughputs = np.zeros(16, dtype=np.float32)

        for link in range(tx_matrices.shape[0]):
            active = np.argwhere(tx_matrices[link] == 1)

            for ap, sta in active:
                sta_idx = int(sta) - 4
                rate = band_rates[link][ap]

                # bits/s -> Mbps
                sta_throughputs[sta_idx] += float(rate) / 1e6

        # --------------------------------
        # Instantaneous Jain Fairness
        # --------------------------------
        numerator = np.sum(sta_throughputs) ** 2
        denominator = 16 * np.sum(sta_throughputs ** 2)

        jain = (
            numerator / denominator
            if denominator > 0
            else 0.0
        )

        # --------------------------------
        # Store results for this step
        # --------------------------------
        throughputs[step] = data_rate
        sta_throughputs_history[step] = sta_throughputs
        fairness_history[step] = jain

        previous_throughput = data_rate

    # --------------------------------
    # Save logger
    # --------------------------------
    logger.save(directory=f"logs/{scenario.str_repr}")

    return (
        throughputs,
        sta_throughputs_history,
        fairness_history
    )

def run_experiments(scenario, args):
    agent_factory = create_agent_factory(scenario, args)
    run_keys = jax.random.split(
        jax.random.PRNGKey(args.seed),
        args.n_runs
    )

    throughputs = np.zeros(
        (args.n_runs, args.n_steps),
        dtype=np.float32,
    )

    sta_throughputs = np.zeros(
        (args.n_runs, args.n_steps, 16),
        dtype=np.float32,
    )

    fairness = np.zeros(
        (args.n_runs, args.n_steps),
        dtype=np.float32,
    )

    for run_number in tqdm(
        range(args.n_runs),
        desc="Running experiments"
    ):
        (
            run_throughputs,
            run_sta_throughputs,
            run_fairness,
        ) = run_single_experiment(
            agent_factory=agent_factory,
            scenario=scenario,
            run_number=run_number,
            n_steps=args.n_steps,
            key=run_keys[run_number],
        )

        throughputs[run_number] = run_throughputs
        sta_throughputs[run_number] = run_sta_throughputs
        fairness[run_number] = run_fairness

    return throughputs, sta_throughputs, fairness


def main():
    args = parse_args()
    Path(args.output_dir).mkdir(
        parents=True,
        exist_ok=True
    )


    jax.config.update("jax_compilation_cache_dir", "./jax_cache")
    jax.config.update("jax_persistent_cache_enable_xla_caches", "all")

    filename = args.filename
    if filename is None:
        filename = (
            f"d_ap_{args.d_ap:g}_"
            f"d_sta_{args.d_sta:g}_"
            f"runs_{args.n_runs}_"
            f"steps_{args.n_steps}"
        )

    scenario = small_office_scenario(
        d_ap=args.d_ap,
        d_sta=args.d_sta,
    )

    throughputs, sta_throughputs, fairness = run_experiments(
        scenario=scenario,
        args=args,
    )

    # --------------------------------------------------
    # 1. Time-averaged throughput for each STA and run
    # --------------------------------------------------

    run_avg_sta = np.mean(
        sta_throughputs[:, 1:, :],
        axis=1
    )

    # --------------------------------------------------
    # 2. Jain fairness for each independent run
    # --------------------------------------------------

    run_jain = []

    for avg_sta in run_avg_sta:
        numerator = np.sum(avg_sta) ** 2
        denominator = 16 * np.sum(avg_sta ** 2)

        jain = (
            numerator / denominator
            if denominator > 0
            else 0.0
        )

        run_jain.append(jain)

    run_jain = np.array(run_jain)

    # --------------------------------------------------
    # 3. Print per-run results
    # --------------------------------------------------

    print("\nPer-run Jain Fairness:")

    for i, jain in enumerate(run_jain):
        print(f"Run {i + 1}: {jain:.4f}")

    print("\nJain Fairness Summary:")
    print(f"Mean: {np.mean(run_jain):.4f}")

    if len(run_jain) > 1:
        print(f"Std:  {np.std(run_jain, ddof=1):.4f}")

    print(
        f"Min:  {np.min(run_jain):.4f}"
    )

    print(
        f"Max:  {np.max(run_jain):.4f}"
    )

    # --------------------------------------------------
    # 4. Average STA throughput across runs
    # --------------------------------------------------

    mean_sta = np.mean(run_avg_sta, axis=0)

    if args.n_runs > 1:
        std_sta = np.std(
            run_avg_sta,
            axis=0,
            ddof=1
        )
    else:
        std_sta = np.zeros(16)

    print("\nAverage STA Throughput Across Runs:")

    for i in range(16):
        print(
            f"STA {i + 4}: "
            f"{mean_sta[i]:.2f} ± {std_sta[i]:.2f} Mbps"
        )

    # --------------------------------------------------
    # Plot 1: Time-averaged throughput per STA
    # --------------------------------------------------

    sta_ids = np.arange(4, 20)

    plt.figure(figsize=(11, 5))

    plt.bar(
        sta_ids,
        mean_sta,
        yerr=std_sta,
        capsize=4
    )

    plt.xlabel("STA")
    plt.ylabel("Average Throughput (Mbps)")
    plt.title("Time-Averaged Throughput per STA")
    plt.xticks(sta_ids)

    plt.tight_layout()

    plt.savefig(
        f"{args.output_dir}/sta_throughput.png",
        dpi=300
    )

    plt.show()

    # --------------------------------------------------
    # 5. Instantaneous Jain
    # --------------------------------------------------

    mean_instantaneous_jain = np.mean(
        fairness[:, 1:]
    )

    print(
        "\nMean Instantaneous Jain Fairness: "
        f"{mean_instantaneous_jain:.4f}"
    )
    
    # --------------------------------------------------
    # Plot 2: Instantaneous Jain fairness over time
    # --------------------------------------------------

    mean_fairness = np.mean(
        fairness[:, 1:],
        axis=0
    )

    std_fairness = np.std(
        fairness[:, 1:],
        axis=0,
        ddof=1
    )

    steps = np.arange(1, args.n_steps)

    plt.figure(figsize=(11, 5))

    plt.plot(
        steps,
        mean_fairness,
        label="Mean Jain Fairness"
    )

    plt.fill_between(
        steps,
        mean_fairness - std_fairness,
        mean_fairness + std_fairness,
        alpha=0.2
    )

    plt.xlabel("Simulation Step")
    plt.ylabel("Jain's Fairness Index")
    plt.title("Instantaneous Jain Fairness Across Simulation Steps")

    plt.ylim(0, 1.05)

    plt.legend()
    plt.subplots_adjust(
    left=0.10,
    right=0.95,
    bottom=0.15,
    top=0.90
)

    plt.savefig(
        f"{args.output_dir}/jain_fairness_vs_step.png",
        dpi=300
    )

    plt.show()

if __name__ == "__main__":       
    main()