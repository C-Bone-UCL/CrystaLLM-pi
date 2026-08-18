"""Broaden XRD peaks and generate debug plots for prefix conditioning.

Inspired by: https://github.com/FrederikLizakJohansen/deCIFer/tree/main
"""


import torch

# Q-space grid: fixed constants for architectural stability
# (changing these would break the first encoder layer input dim)
QMIN = 0.0
QMAX = 10.0
QSTEP = 0.01
NUM_Q_POINTS = int((QMAX - QMIN) / QSTEP)  # 1000 points

# Default broadening ranges
FWHM_RANGE = (0.01, 0.5)
ETA_RANGE = (0.5, 0.5)


def condition_vector_to_continuous_xrd(
    condition_vector: object,
    seed: int = 1,
    fwhm: float = 0.05,
    eta: float = 0.5,
) -> list[list[float]]:
    """Convert a condition vector into the 1000x2 `[Q, I]` form PrefixXRD consumes.

    Accepts either discrete `[Q, I]` peaks, which are broadened, or an already-continuous 1000x2 profile, which is passed through once its Q column is confirmed to match the canonical grid. Broadening here is deterministic: noise, rescaling and masking are all disabled and `fwhm` and `eta` are fixed.
    """
    values = condition_vector.tolist() if hasattr(condition_vector, "tolist") else condition_vector
    if not values:
        return []
    values = [row.tolist() if hasattr(row, "tolist") else row for row in values]

    tensor = torch.tensor(values, dtype=torch.float32)
    q_grid = torch.arange(QMIN, QMAX, QSTEP, dtype=torch.float32)

    if tensor.ndim != 2 or tensor.shape[1] != 2:
        raise ValueError("condition_vector must be discrete [Q, I] peaks or continuous 1000x2 [Q, I] points")

    if tensor.shape[0] == NUM_Q_POINTS and torch.allclose(tensor[:, 0], q_grid, atol=1e-4, rtol=0.0):
        return tensor.tolist()

    xrd = discrete_to_continuous_xrd(
        batch_q=tensor[:, 0].unsqueeze(0),
        batch_iq=tensor[:, 1].unsqueeze(0),
        fwhm_range=(fwhm, fwhm),
        eta_range=(eta, eta),
        noise_range=None,
        intensity_scale_range=None,
        mask_prob=None,
        seed=seed,
    )
    return torch.stack([xrd["q"], xrd["iq"][0]], dim=1).cpu().tolist()


def discrete_to_continuous_xrd(
    batch_q: torch.Tensor,
    batch_iq: torch.Tensor,
    qmin: float = QMIN,
    qmax: float = QMAX,
    qstep: float = QSTEP,
    fwhm_range: tuple[float, float] = FWHM_RANGE,
    eta_range: tuple[float, float] = ETA_RANGE,
    noise_range: tuple[float, float] | None = (0.001, 0.05),
    intensity_scale_range: tuple[float, float] | None = (0.95, 1.0),
    mask_prob: float | None = None,
    seed: int | None = None,
    **kwargs,
) -> dict:
    """Broaden discrete XRD peaks into a continuous 1000-point profile.

    The augmentation args exist for training. Inference disables noise, scaling and masking and pins fwhm and eta, so the same peaks always give the same profile.

    Args:
        batch_q: [B, N_peaks] - peak positions in A^-1, Q == 0 treated as padding
        batch_iq: [B, N_peaks] - peak intensities
        qmin: grid lower bound in A^-1, default 0.0
        qmax: grid upper bound in A^-1, default 10.0
        qstep: grid spacing in A^-1, default 0.01
        fwhm_range: peak width sampled per batch item, in A^-1
        eta_range: pseudo-Voigt mixing, 0 Gaussian to 1 Lorentzian
        noise_range: noise amplitude, None disables
        intensity_scale_range: random rescaling, None disables
        mask_prob: per-peak drop probability, None disables
        seed: seed for the augmentation draws

    Returns:
        dict with q [1000] shared grid and iq [B, 1000] max-normalized to [0, 1]
    """
    device = batch_q.device
    generator = None
    if seed is not None:
        generator = torch.Generator(device=device.type)
        generator.manual_seed(int(seed))

    def sample(shape: tuple[int, ...], value_range: tuple[float, float]) -> torch.Tensor:
        return torch.empty(shape, device=device).uniform_(*value_range, generator=generator)

    q_cont = torch.arange(qmin, qmax, qstep, device=device)
    batch_size = batch_q.shape[0]
    num_q_points = q_cont.shape[0]

    fwhm = sample((batch_size, 1, 1), fwhm_range)
    eta = sample((batch_size, 1, 1), eta_range)

    if intensity_scale_range is not None:
        intensity_scale = sample((batch_size, 1), intensity_scale_range)
        batch_iq = batch_iq * intensity_scale

    sigma_gauss = fwhm / (2 * torch.sqrt(2 * torch.log(torch.tensor(2.0, device=device))))
    gamma_lorentz = fwhm / 2

    q_cont_expanded = q_cont.view(1, num_q_points, 1)
    batch_q_expanded = batch_q.unsqueeze(1)
    delta_q = q_cont_expanded - batch_q_expanded

    gaussian_component = torch.exp(-0.5 * (delta_q / sigma_gauss) ** 2)
    lorentzian_component = 1 / (1 + (delta_q / gamma_lorentz) ** 2)
    pseudo_voigt = eta * lorentzian_component + (1 - eta) * gaussian_component

    batch_iq_expanded = batch_iq.unsqueeze(1)
    # Q == 0 is reserved for clamped padding rows, not real peaks.
    valid_peaks = (batch_q_expanded != 0).float()
    iq_cont = (pseudo_voigt * batch_iq_expanded * valid_peaks).sum(dim=2)
    iq_cont /= (iq_cont.max(dim=1, keepdim=True)[0] + 1e-16)

    if noise_range is not None:
        noise_scale = sample((batch_size, 1), noise_range)
        noise = torch.randn(batch_size, num_q_points, device=device, generator=generator)
        iq_cont = iq_cont + noise * noise_scale

    iq_cont = torch.clamp(iq_cont, min=0.0)

    # mask_prob from deCIFer is disabled here.
    # if mask_prob is not None:
    #     mask = (torch.rand(batch_size, num_q_points, device=device) > mask_prob).float()
    #     iq_cont *= mask

    return {'q': q_cont, 'iq': iq_cont}


def save_xrd_pipeline_plots(
    batch_q: torch.Tensor | None,
    batch_iq: torch.Tensor | None,
    q_cont: torch.Tensor,
    iq_cont: torch.Tensor,
    save_dir: str,
    step: int = 0,
    max_examples: int = 2,
    figure_size: tuple[float, float] = (3.45, 3.8),
) -> list[str]:
    """Save PrefixXRD pipeline plots for manual inspection and return their paths."""
    import os
    import pathlib
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
    batch_size = iq_cont.shape[0]
    n_examples = min(max_examples, batch_size)
    q_np = q_cont.cpu().float().numpy()
    saved_paths = []
    peak_color = "#2f6fbb"
    profile_color = "#d16621"
    point_color = "#2f8f4e"

    for i in range(n_examples):
        iq_np = iq_cont[i].cpu().float().numpy()

        peaks_q = np.array([])
        peaks_iq = np.array([])
        if batch_q is not None and batch_iq is not None:
            peaks_q = batch_q[i].cpu().float().numpy()
            peaks_iq = batch_iq[i].cpu().float().numpy()
            valid = peaks_q > 0
            peaks_q = peaks_q[valid]
            peaks_iq = peaks_iq[valid]

        fig, (ax_top, ax_mid, ax_bot) = plt.subplots(
            3,
            1,
            figsize=figure_size,
            sharex=True,
            gridspec_kw={"height_ratios": [0.72, 1.0, 1.0], "hspace": 0.12},
        )
        fig.suptitle("Synthetic XRD conditioning example", fontsize=8.6, y=0.985)

        if len(peaks_q) > 0:
            ax_top.vlines(peaks_q, 0.0, peaks_iq, color=peak_color, linewidth=0.85)
            ax_top.scatter(peaks_q, peaks_iq, s=10.0, color=peak_color, linewidths=0, label="Discrete peaks")
            peak_upper = max(1.05, float(np.nanmax(peaks_iq)) * 1.12)
        else:
            ax_top.plot(q_np, iq_np, color=peak_color, linewidth=1.15, label="Input profile")
            peak_upper = max(1.05, float(np.nanmax(iq_np)) * 1.08)
        ax_top.axhline(0.0, color="black", linewidth=0.70)
        ax_top.set_ylim(-0.035 * peak_upper, peak_upper)

        ax_mid.plot(q_np, iq_np, color=profile_color, linewidth=1.25, label="Broadened profile")
        profile_upper = max(1.05, float(np.nanmax(iq_np)) * 1.08)
        ax_mid.set_ylim(-0.035 * profile_upper, profile_upper)

        ax_bot.scatter(
            q_np,
            iq_np,
            s=3.0,
            color=point_color,
            alpha=0.62,
            linewidths=0,
            label="Conditioning points",
        )
        ax_bot.set_ylim(-0.035 * profile_upper, profile_upper)
        ax_bot.set_xlabel(r"Q ($\mathrm{\AA}^{-1}$)", fontsize=7.8, labelpad=1.8)

        for ax in (ax_top, ax_mid, ax_bot):
            ax.set_xlim(0, 10)
            ax.grid(axis="y", alpha=0.22, linewidth=0.55)
            ax.tick_params(axis="y", labelsize=7.2, length=2.5, pad=1.5)
            ax.tick_params(axis="x", labelsize=7.2, length=2.5, pad=1.5)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("bottom", "left"):
                ax.spines[side].set_linewidth(0.75)

        legend_handles = []
        legend_labels = []
        for ax in (ax_top, ax_mid, ax_bot):
            handles, labels = ax.get_legend_handles_labels()
            for handle, label in zip(handles, labels):
                if label not in legend_labels:
                    legend_handles.append(handle)
                    legend_labels.append(label)
        fig.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.006),
            ncol=min(len(legend_handles), 3),
            frameon=False,
            fontsize=7.1,
            handlelength=1.35,
            columnspacing=0.82,
        )
        fig.supylabel("Relative intensity", fontsize=7.8, x=0.030)
        fig.subplots_adjust(left=0.165, right=0.995, bottom=0.185, top=0.905)

        out_path = os.path.join(save_dir, f"step{step:05d}_example{i+1}.png")
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(out_path)

    return saved_paths


def xrd_debug_check(
    batch_q: torch.Tensor | None,
    batch_iq: torch.Tensor | None,
    q_cont: torch.Tensor,
    iq_cont: torch.Tensor,
    past_key_values: tuple,
    config: object,
    step: int = 0,
    save_dir: str = "debug_xrd",
) -> None:
    """Run PrefixXRD debug checks and save pipeline plots."""
    batch_size = iq_cont.shape[0]
    print(f"\n[XRD DEBUG] step={step}  batch_size={batch_size}")

    assert iq_cont.shape == (batch_size, 1000), f"iq_cont: {iq_cont.shape} != ({batch_size}, 1000)"
    print(f"  [PASS] continuous spectrum : {tuple(iq_cont.shape)}")

    n_layer = config.n_layer
    n_head = config.n_head
    n_prefix = config.n_prefix_tokens
    head_dim = config.hidden_size // config.n_head

    assert len(past_key_values) == n_layer, \
        f"PKV len {len(past_key_values)} != n_layer {n_layer}"
    print(f"  [PASS] PKV layers : {len(past_key_values)}/{n_layer}")

    exp_shape = (batch_size, n_head, n_prefix, head_dim)
    k0, v0 = past_key_values[0]
    assert k0.shape == exp_shape, f"K[0] shape {tuple(k0.shape)} != {exp_shape}"
    assert v0.shape == exp_shape, f"V[0] shape {tuple(v0.shape)} != {exp_shape}"
    print(f"  [PASS] PKV[0] K : {tuple(k0.shape)}")
    print(f"  [PASS] PKV[0] V : {tuple(v0.shape)}")

    for l_idx, (k_l, v_l) in enumerate(past_key_values):
        assert k_l.shape == exp_shape, f"K[{l_idx}] shape mismatch: {tuple(k_l.shape)}"
        assert v_l.shape == exp_shape, f"V[{l_idx}] shape mismatch: {tuple(v_l.shape)}"
    print(f"  [PASS] All PKV layers shape-consistent")

    nan_iq = torch.isnan(iq_cont).any().item()
    inf_iq = torch.isinf(iq_cont).any().item()
    assert not nan_iq, "NaN detected in continuous XRD spectrum!"
    assert not inf_iq, "Inf detected in continuous XRD spectrum!"
    print(f"  [PASS] I(Q) range: [{iq_cont.min():.4f}, {iq_cont.max():.4f}]  NaN={nan_iq}")

    assert abs(q_cont[0].item() - 0.0) < 1e-3, f"Q grid starts at {q_cont[0]:.4f}, expected ~0.0"
    assert abs(q_cont[-1].item() - 9.99) < 0.02, f"Q grid ends at {q_cont[-1]:.4f}, expected ~9.99"
    print(f"  [PASS] Q grid: [{q_cont[0]:.3f}, {q_cont[-1]:.3f}]  ({q_cont.shape[0]} points)")

    pkv_k_all = torch.stack([past_key_values[i][0] for i in range(n_layer)])
    nan_k = torch.isnan(pkv_k_all).any().item()
    assert not nan_k, "NaN in PKV K tensors - check encoder numerical stability!"
    print(f"  [PASS] PKV K: mean={pkv_k_all.mean():.4f}  std={pkv_k_all.std():.4f}  max_abs={pkv_k_all.abs().max():.4f}")

    if pkv_k_all.abs().max().item() > 100:
        print(f"  [WARN] PKV K max_abs > 100 - encoder might not be trained yet")

    saved_paths = save_xrd_pipeline_plots(
        batch_q=batch_q,
        batch_iq=batch_iq,
        q_cont=q_cont,
        iq_cont=iq_cont,
        save_dir=save_dir,
        step=step,
    )
    for out_path in saved_paths:
        print(f"  Saved: {out_path}")

    print(f"[XRD DEBUG] All checks passed.\n")
