import platform
import os
import matplotlib.pyplot as plt
import wandb

from matplotlib.ticker import MultipleLocator


COLORS = ['#D96248', '#554687', '#F2A81D', '#285947', '#9B41D1', '#72DB1A', '#342B56', '#4A96D9', '#F22E8A', '#FF0200']
MARKERS = ['8', 's', 'p', '<', 'D', '>', 'P', 'h', '*', 'X']


def _resolve_wandb_project_path(project_name: str) -> str:
    if "/" in project_name:
        return project_name
    entity = os.environ.get("WANDB_ENTITY") or wandb.Api().viewer.teams[-1]
    if not entity:
        raise ValueError(
            "Could not infer W&B entity. Pass project as 'entity/project' "
            "or set WANDB_ENTITY."
        )
    return f"{entity}/{project_name}"


def _moving_average(values: list[float], window: int) -> list[float]:
    if window <= 1:
        return values
    window = min(window, len(values))
    out: list[float] = []
    running = 0.0
    for i, v in enumerate(values):
        running += v
        if i >= window:
            running -= values[i - window]
            out.append(running / window)
        else:
            out.append(running / (i + 1))
    return out


def _get_train_reward_path(run_name: str) -> str:
    model_name, dataset_name, rag_method = run_name.split("|")
    model_size = model_name.split("-")[-1].lower()
    dataset_name=dataset_name.replace("_filtered", "")
    rag_method=rag_method.replace("_", "-")
    return f"figures/train_reward_{model_size}_{dataset_name}_{rag_method}.pdf"


def _get_grad_norm_path(run_names: list[str]) -> str:
    model_name, dataset_name, rag_method = run_names[0].split("|")[:3]
    model_size = model_name.split("-")[-1].lower()
    dataset_name=dataset_name.replace("_filtered", "")
    return f"figures/grad_norm_{model_size}_{dataset_name}.pdf"


def plot_wandb_train_reward_raw(
    project_name: str,
    run_name: str,
    smooth_window: int = 1,
    save: bool = False,
):
    """
    Download `train/reward_raw` from W&B and plot it as a line chart.

    Args:
        project_name: W&B project name or 'entity/project'.
        run_name: W&B run name (matches `run.name` or `run.display_name`).
        smooth_window: Moving-average window size. 1 disables smoothing.

    Returns:
        (fig, ax) matplotlib objects.
    """
    project_path = _resolve_wandb_project_path(project_name)
    api = wandb.Api()

    target_run = None
    for run in api.runs(project_path):
        if run.name == run_name or getattr(run, "display_name", None) == run_name:
            target_run = run
            break

    if target_run is None:
        raise ValueError(f"Run '{run_name}' not found in project '{project_path}'.")

    rewards: list[float] = []
    for row in target_run.scan_history(keys=["train/reward_raw"], page_size=10_000):
        reward = row.get("train/reward_raw")
        if reward is None:
            continue
        rewards.append(float(reward))

    if not rewards:
        raise ValueError("No 'train/reward_raw' points found for this run.")

    rewards_plot = _moving_average(rewards, smooth_window)
    steps = list(range(1, len(rewards_plot) + 1))

    fig, ax = plt.subplots()
    color = COLORS[0]
    ax.plot(steps, rewards_plot, color=color, linewidth=1.5)
    ax.scatter(
        steps,
        rewards_plot,
        s=60,
        marker="o",
        facecolors=color,
        edgecolors="none",
        zorder=3,
    )
    ax.set_xlabel("Step", fontsize=30)
    ax.set_ylabel("Train Reward", fontsize=30)
    ax.xaxis.set_major_locator(MultipleLocator(25))
    ax.tick_params(labelsize=30)
    ax.grid(True, alpha=0.25, linewidth=0.8)
    fig.tight_layout()

    fig.show()
    if save:
        os.makedirs("figures", exist_ok=True)
        save_path = _get_train_reward_path(run_name)
        fig.savefig(save_path, backend='pgf')
        print(f"Saved plot to {save_path}")

    return fig, ax


def plot_wandb_train_grad_norm_area(
    project_name: str,
    run_names: list[str],
    save: bool = False,
):
    """
    Download `train/grad_norm` for multiple runs from W&B and plot them in one area chart.

    Args:
        project_name: W&B project name or 'entity/project'.
        run_names: List of W&B run names (matches `run.name` or `run.display_name`).
        save: Whether to save the figure under `figures/`.

    Returns:
        (fig, ax) matplotlib objects.
    """
    if not run_names:
        raise ValueError("run_names must be non-empty.")

    project_path = _resolve_wandb_project_path(project_name)
    api = wandb.Api()

    run_by_name = {}
    for run in api.runs(project_path):
        run_by_name[run.name] = run
        display_name = getattr(run, "display_name", None)
        if display_name:
            run_by_name[display_name] = run

    fig, ax = plt.subplots()

    for i, run_name in enumerate(run_names):
        run = run_by_name.get(run_name)
        if run is None:
            raise ValueError(f"Run '{run_name}' not found in project '{project_path}'.")

        steps: list[int] = []
        grad_norms: list[float] = []
        for row in run.scan_history(keys=["train/grad_norm"], page_size=10_000):
            grad = row.get("train/grad_norm")
            if grad is None:
                continue
            step = row.get("_step")
            grad_norms.append(float(grad))
            steps.append(int(step) if step is not None else len(grad_norms))

        if not grad_norms:
            raise ValueError(f"No 'train/grad_norm' points found for run '{run_name}'.")

        paired = sorted(zip(steps, grad_norms), key=lambda x: x[0])
        steps = [p[0] for p in paired]
        grad_norms = [p[1] for p in paired]

        color = COLORS[i % len(COLORS)]
        ax.fill_between(steps, grad_norms, color=color, alpha=0.25, linewidth=0)
        ax.plot(steps, grad_norms, color=color, linewidth=1.5,
                label="w/o History" if "ablation" in run_name else "HARR")

    ax.set_xlabel("Step", fontsize=30)
    ax.set_ylabel("Grad Norm", fontsize=30)
    ax.tick_params(labelsize=30)
    ax.grid(True, alpha=0.25, linewidth=0.8)
    fig.tight_layout()

    fig.show()
    if save:
        os.makedirs("figures", exist_ok=True)
        save_path = _get_grad_norm_path(run_names)
        fig.savefig(save_path, backend="pgf")
        print(f"Saved plot to {save_path}")

        handles, labels = ax.get_legend_handles_labels()
        if handles and labels:
            legend_fig = plt.figure()
            legend_ax = legend_fig.add_subplot(111)
            legend_ax.axis("off")
            legend = legend_ax.legend(
                handles,
                labels,
                loc="center",
                frameon=True,
                fancybox=True,
                fontsize=26,
                ncol=len(labels),
            )
            legend_path = save_path.rsplit(".", 1)[0] + "_legend." + save_path.rsplit(".", 1)[1]
            # Resize figure to the legend's exact bounding box and save with no padding.
            legend_fig.canvas.draw()
            bbox = legend.get_window_extent(legend_fig.canvas.get_renderer())
            bbox = bbox.transformed(legend_fig.dpi_scale_trans.inverted())
            legend_fig.set_size_inches(bbox.width, bbox.height)
            legend_fig.savefig(
                legend_path,
                backend="pgf",
                bbox_inches="tight",
                pad_inches=0.1,
            )
            print(f"Saved legend to {legend_path}")
            plt.close(legend_fig)

    return fig, ax


if __name__ == "__main__":
    zh_font = "Songti SC" if platform.system() == "Darwin" else "SimSun"
    config = {
        "font.family": "serif",
        "font.serif": [
            "Times New Roman",
            zh_font,
        ],  # Times New Roman for en，SimSun for zh
        "font.size": 14,
        "font.weight": "normal",
        "axes.unicode_minus": False,
        "mathtext.fontset": "stix",  # a font similar to Times New Roman, but with math symbols
    }
    plt.rcParams.update(config)

    # Plot Figure 2
    run_names = [
        "Qwen3-Embedding-4B|hotpotqa_filtered|react_agent",
        "Qwen3-Embedding-4B|hotpotqa_filtered|search_r1",
        "Qwen3-Embedding-0.6B|hotpotqa_filtered|react_agent",
        "Qwen3-Embedding-0.6B|hotpotqa_filtered|search_r1",
    ]
    for run_name in run_names:
        plot_wandb_train_reward_raw(
            project_name="RAG-RL-GRPO",
            run_name=run_name,
            smooth_window=8,
            save=True,
        )

    # Plot Figure 3
    run_names = [
        ["Qwen3-Embedding-4B|hotpotqa_filtered|react_agent",
         "Qwen3-Embedding-4B|hotpotqa_filtered|react_agent|ablation"],
        ["Qwen3-Embedding-0.6B|hotpotqa_filtered|react_agent",
         "Qwen3-Embedding-0.6B|hotpotqa_filtered|react_agent|ablation"]
    ]
    for rn in run_names:
        plot_wandb_train_grad_norm_area(
            project_name="RAG-RL-GRPO",
            run_names=rn,
            save=True,
        )
