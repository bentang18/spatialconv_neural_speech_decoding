"""EMA teacher + explicit StopGrad (T2.2) + phase-aware schedules + layer-avg target.

Recipe-independent — used by the single joint SSL phase (B29 merged the old
Phase-1 + Phase-2), and available for Phase-3 if a teacher-side EMA target is
added. The ``p1_``/``p2_`` schedule helpers below predate the merge; they are
retained only as fixed-τ aliases / sister-sweep primitives.

``EmaTeacher`` keeps a copy of the student under ``no_grad`` and updates it in
place after each optimizer step:

    teacher.param ← coeff · teacher.param + (1 − coeff) · student.param

The coefficient schedule is callable (``coeff(step) → float``).

**B26 amendment 2026-05-27 PM; value corrected 2026-05-30**: EMA momentum
is locked at **fixed τ=0.99925** throughout the joint SSL phase. The
fixed/no-ramp form is V-JEPA 2 §2.4 (which explicitly drops the V-JEPA
1-style 0.996 → 1.0 ramp); the *value* 0.99925 is V-JEPA 2's disclosed
pretraining EMA (main.tex:957) and is identical in V-JEPA 2.1 (App. A).
The earlier B26 value τ=0.999 was set on the false premise that V-JEPA 2
doesn't disclose its τ — it does; 0.999 is only the *start* of a *ramped*
schedule in V-JEPA 2's abbreviated 90k-step ablation recipe. The previous
ramping schedules (P1 0.99 → 0.9999 over 400k steps, P2 0.999 → 0.9999
over 40k steps) are retired. The V-JEPA 1 ramp is preserved as the
``R-ema-ramp-v-jepa1`` sister-cell falsifier via
:func:`v_jepa1_ema_schedule`. Custom τ sweeps (``R-ema-tau-{0.99, 0.9995,
0.9999}``) use :func:`fixed_ema_schedule(tau=...)` directly.

P3 distillation uses an external frozen Whisper teacher and skips EMA entirely.

``layer_avg_with_instance_norm()`` (data2vec-2.0 / EAT recipe layer-averaging
with per-layer instance-norm; accepts K=6 layer outputs, returns the per-layer
instance-normed average) is **RETIRED from the default path (B36 WS-B / B9)**.
Under the canonical V-JEPA-2 target-norm the SSL target is the EMA teacher's
own terminal LayerNorm tap (``frontend_ln`` at M2, ``encoder_ln`` at M4), so
this helper builds NO live target — it is on no runtime path (grep confirms:
only this module + its unit test reference it). It is kept solely as the
``R-layer-avg-target`` data2vec-recipe sister-cell falsifier and its numeric
unit test; the earlier docstring claim that it is "used by ``L_post_frame`` /
the ``L_mid_slot`` sister" is stale and was corrected here.

**B26 teacher full-input contract** (V-JEPA 2 §2.1, V-JEPA 2.1 §2.3.1):
in the SSL loss path, the EMA teacher MUST encode the full unmasked
input — no patch drop, no shaft drop. Asymmetric supervision (student
sees masked input, teacher sees full input) IS the JEPA loss signal.
:func:`assert_teacher_full_input` is a cheap debug assert that catches
contract violations at the teacher-forward call site.
"""

from __future__ import annotations

import copy
import typing as tp

import torch
from torch import Tensor, nn


# Step counts pinned in training_recipe.md §3 (P1 ~ 400k) and §4 (P2 ~ 40k).
# Retained for callers that still derive ramp-style sweeps from them (e.g.
# the R-ema-ramp-v-jepa1 sister); the B26 default τ does not depend on
# total_steps because it's a fixed scalar.
P1_TOTAL_STEPS: int = 400_000
P2_TOTAL_STEPS: int = 40_000

# B26 lock 2026-05-27 PM: fixed EMA momentum across P1 and P2 per
# V-JEPA 2 §2.4. The previous P1 (0.99 → 0.9999) and P2 (0.99925 → 0.9999)
# linear ramps are retired.
P1_EMA_TAU: float = 0.99925
P2_EMA_TAU: float = 0.99925


def stop_grad(x: Tensor) -> Tensor:
    """Explicit StopGrad wrapper. Named so future readers can grep for the
    operation rather than tracking the meaning of bare ``.detach()`` calls
    sprinkled through the SSL loss code."""
    return x.detach()


def fixed_ema_schedule(tau: float = 0.99925) -> tp.Callable[[int], float]:
    """Constant EMA momentum (B26 lock).

    Returns ``coeff(step) → tau`` for all ``step``. ``tau=0.99925`` is
    V-JEPA 2's disclosed pretraining EMA (main.tex:957; fixed/no-ramp per
    §2.4; identical in V-JEPA 2.1 App. A); the ``R-ema-tau-{0.99, 0.9995, 0.9999}``
    sister sweep wires custom values.
    """
    if not 0.0 < tau < 1.0:
        raise ValueError(f"tau must be in (0.0, 1.0), got {tau}")

    def coeff(_step: int) -> float:
        return float(tau)

    return coeff


def linear_ema_schedule(
    start: float = 0.99,
    end: float = 0.9999,
    total_steps: int = 10_000,
) -> tp.Callable[[int], float]:
    """Linear ramp from ``start`` to ``end`` over ``total_steps`` calls.
    Steps past ``total_steps`` clamp at ``end``.

    Retained for custom EMA sweeps and as the underlying primitive of the
    :func:`v_jepa1_ema_schedule` sister-cell falsifier. **Not** used by
    the B26 default — :func:`p1_ema_schedule` and :func:`p2_ema_schedule`
    now return :func:`fixed_ema_schedule(0.99925)`.
    """
    if total_steps <= 0:
        raise ValueError(f"total_steps must be positive, got {total_steps}")

    def coeff(step: int) -> float:
        t = min(max(step, 0), total_steps) / float(total_steps)
        return start + (end - start) * t

    return coeff


def p1_ema_schedule(
    total_steps: int = P1_TOTAL_STEPS,  # noqa: ARG001  retained for signature stability
) -> tp.Callable[[int], float]:
    """Phase-1 EMA momentum (B26 lock 2026-05-27 PM): fixed τ=0.99925
    (V-JEPA 2 main.tex:957; fixed/no-ramp per §2.4).

    The ``total_steps`` parameter is retained for backward-compatible
    callers but ignored — the schedule is a constant.
    """
    return fixed_ema_schedule(tau=P1_EMA_TAU)


def p2_ema_schedule(
    total_steps: int = P2_TOTAL_STEPS,  # noqa: ARG001  retained for signature stability
) -> tp.Callable[[int], float]:
    """Phase-2 EMA momentum (B26 lock 2026-05-27 PM): fixed τ=0.99925
    (V-JEPA 2 main.tex:957; fixed/no-ramp per §2.4).

    The ``total_steps`` parameter is retained for backward-compatible
    callers but ignored — the schedule is a constant.
    """
    return fixed_ema_schedule(tau=P2_EMA_TAU)


def v_jepa1_ema_schedule(
    total_steps: int = P1_TOTAL_STEPS,
) -> tp.Callable[[int], float]:
    """``R-ema-ramp-v-jepa1`` sister cell (B26): V-JEPA 1-style ramp
    0.996 → 1.0 over ``total_steps``. Falsifies V-JEPA 2's drop-the-ramp
    decision at v14 scale.

    Not the v14 default — kept callable so the dispatch flag can wire it
    in for the sister run.
    """
    return linear_ema_schedule(start=0.996, end=1.0, total_steps=total_steps)


def phase_ema_schedule(
    phase: tp.Literal[1, 2],
    *,
    total_steps: int | None = None,  # noqa: ARG001  retained for signature stability
) -> tp.Callable[[int], float]:
    """Dispatch helper for V14Experiment: returns the right schedule for
    the phase. Phase 3 is intentionally not supported — distillation uses
    an external frozen Whisper teacher, no EMA.

    Under the B26 lock, both P1 and P2 return :func:`fixed_ema_schedule`
    at τ=0.99925. ``total_steps`` is accepted for backward compatibility
    and ignored.
    """
    if phase == 1:
        return p1_ema_schedule()
    if phase == 2:
        return p2_ema_schedule()
    raise ValueError(
        f"phase_ema_schedule supports phases 1 and 2 only; "
        f"got phase={phase!r} (P3 distillation uses external Whisper teacher, no EMA)"
    )


def assert_teacher_full_input(
    *,
    patch_mask: tp.Optional[Tensor] = None,
    shaft_mask: tp.Optional[Tensor] = None,
) -> None:
    """B26 teacher full-input contract debug assert (V-JEPA 2 §2.1,
    V-JEPA 2.1 §2.3.1).

    In the SSL loss path the EMA teacher MUST encode the **full unmasked
    input** — no patch drop, no shaft drop. Asymmetric supervision
    (student sees masked input, teacher sees full input) IS the JEPA
    loss signal. Call this helper at the teacher-forward call site to
    catch contract violations.

    Parameters
    ----------
    patch_mask
        Optional patch-level mask tensor. ``None`` means no patch mask
        was passed (which is the desired contract). A non-``None`` mask
        must be all-True (= every patch visible to the teacher).
    shaft_mask
        Optional shaft-level mask tensor. Same contract: ``None`` is
        ideal; non-``None`` must be all-True.

    Raises
    ------
    AssertionError
        If either mask is non-``None`` and contains any ``False`` entry.

    Notes
    -----
    Cheap debug assert — single ``.all()`` per non-``None`` mask. Safe
    to leave on in the training loop; the cost is negligible compared
    to the teacher forward.
    """
    if patch_mask is not None and not torch.as_tensor(patch_mask).all():
        raise AssertionError(
            "teacher full-input contract violated: patch_mask contains "
            "False entries — the EMA teacher must encode the full "
            "unmasked input (V-JEPA 2 §2.1, V-JEPA 2.1 §2.3.1)."
        )
    if shaft_mask is not None and not torch.as_tensor(shaft_mask).all():
        raise AssertionError(
            "teacher full-input contract violated: shaft_mask contains "
            "False entries — the EMA teacher must encode the full "
            "unmasked input (V-JEPA 2 §2.1, V-JEPA 2.1 §2.3.1)."
        )


def layer_avg_with_instance_norm(
    layer_outputs: tp.Sequence[Tensor],
    *,
    instance_norm_dim: int = -1,
    eps: float = 1e-6,
) -> Tensor:
    """data2vec-2.0 / EAT teacher target builder.

    **RETIRED from the default path (B36 WS-B / B9).** The canonical
    V-JEPA-2 target is the EMA teacher's own terminal LayerNorm tap, so this
    layer-averaging builder is on no runtime path; it survives only as the
    ``R-layer-avg-target`` sister-cell falsifier and its unit test. The
    P1/P2 "symmetric across N=6 token blocks / L=6 latent stack" note below
    describes that retired data2vec recipe, NOT the live masked-JEPA loss.

    For each layer output ``L_k`` (shape ``(*lead, d)``), apply instance-norm
    across ``instance_norm_dim`` then average across the K layer outputs.
    Guards against EMA-self-teacher collapse by normalizing per-layer before
    averaging — without this, deeper layers dominate the average through
    larger activation magnitudes.

    Use the LAST K layers from the teacher's forward (typically K=6 for v14's
    L=6 latent stack); caller selects which K.
    """
    if len(layer_outputs) == 0:
        raise ValueError("layer_outputs must be non-empty")
    normed = []
    for layer in layer_outputs:
        mu = layer.mean(dim=instance_norm_dim, keepdim=True)
        sigma = layer.std(dim=instance_norm_dim, keepdim=True).clamp(min=eps)
        normed.append((layer - mu) / sigma)
    return torch.stack(normed, dim=0).mean(dim=0)


class EmaTeacher(nn.Module):
    """A frozen-by-construction EMA copy of ``student``.

    Usage:

        student = MyModel(...)
        teacher = EmaTeacher(student, coeff_schedule=linear_ema_schedule())
        ...
        # Each train step:
        student_out = student(x_masked)
        with torch.no_grad():
            teacher_out = teacher(x_target)
        loss = recon_loss(student_out, stop_grad(teacher_out), ...)
        loss.backward()
        optim.step()
        teacher.update_from(student)

    Implementation notes:

      * The EMA copy is a deepcopy of the student so the structure tracks
        exactly. Buffers are copied as-is on each ``update_from`` (since EMA
        of running stats is generally not what you want — copy is closer to
        the V-JEPA 2.1 reference).
      * ``requires_grad`` is set to False on every parameter at construction
        AND re-cleared after each ``load_state_dict``-equivalent op, so it
        cannot accidentally accumulate gradients downstream.
      * ``step`` is monotone, advanced by ``update_from``. Callers can pass
        their own step via ``step=`` to override (useful in tests).
    """

    def __init__(
        self,
        student: nn.Module,
        *,
        coeff_schedule: tp.Optional[tp.Callable[[int], float]] = None,
    ) -> None:
        super().__init__()
        self.model = copy.deepcopy(student)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self._coeff_schedule = coeff_schedule or linear_ema_schedule()
        self.register_buffer("_step", torch.tensor(0, dtype=torch.long), persistent=True)

    def forward(self, *args: tp.Any, **kwargs: tp.Any) -> tp.Any:
        # The teacher always runs under no_grad — the wrapper ensures even a
        # forgetful caller can't blow the gradient budget.
        with torch.no_grad():
            return self.model(*args, **kwargs)

    @torch.no_grad()
    def update_from(
        self,
        student: nn.Module,
        *,
        step: tp.Optional[int] = None,
    ) -> float:
        """In-place EMA update from ``student``. Returns the coefficient used."""
        if step is None:
            step = int(self._step.item())
        coeff = float(self._coeff_schedule(step))
        student_state = student.state_dict()
        teacher_state = self.model.state_dict()
        # Dedup by storage address (2026-05-30 speedup audit, #135). A module
        # assigned under two attribute names — e.g. the v14 encoder's legacy
        # ``self.cross_attn = self.cross_attns[0]`` handle — appears in
        # ``state_dict()`` under BOTH keys, each a distinct ``.detach()`` view
        # sharing one storage. Iterating naively applied the EMA twice to those
        # tensors → effective coeff² (τ≈0.9985 not 0.99925), silently breaking the
        # B26 uniform-τ contract for the anatomy-routing cross-attn params. Key
        # on ``data_ptr()`` (storage start + offset) so each unique tensor is
        # updated exactly once; ``state_dict`` returns detached views so
        # in-place ops still write through to the live teacher params.
        seen: set[int] = set()
        ema_t: list[Tensor] = []
        ema_s: list[Tensor] = []
        for name, t_param in teacher_state.items():
            ptr = t_param.data_ptr()
            if ptr in seen:
                continue
            seen.add(ptr)
            s_param = student_state[name]
            if t_param.dtype.is_floating_point and s_param.dtype.is_floating_point:
                ema_t.append(t_param)
                ema_s.append(s_param)
            else:
                # Non-float buffers (e.g. step counters) get copied verbatim.
                t_param.copy_(s_param)
        # Fused multi-tensor EMA: one ``_foreach`` kernel pair for the whole
        # float param set instead of a per-tensor Python ``mul_/add_`` loop.
        # Bit-identical to the sequential loop now that aliases are deduped —
        # each tensor independently becomes ``t*coeff + s*(1-coeff)``.
        if ema_t:
            torch._foreach_mul_(ema_t, coeff)
            torch._foreach_add_(ema_t, ema_s, alpha=1.0 - coeff)
        # Defensive: ensure no parameter accidentally has requires_grad re-set.
        for p in self.model.parameters():
            p.requires_grad_(False)
        self._step += 1
        return coeff
