# ruff: noqa
# mypy: ignore-errors
# pyright: reportUndefinedVariable=false, reportGeneralTypeIssues=false
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except Exception:  # pragma: no cover - platform without Triton
    triton = None
    tl = None
    _TRITON_AVAILABLE = False
import math
import os
import warnings
from functools import lru_cache


class EvoAttentionCausalTorch(nn.Module):
    def __init__(self):
        super().__init__()
        # Lazy initialization to keep API backward-compatible (no required d_model arg)
        self._initialized = False
        # Use Optional typing-compatible with py310 tooling
        self.q_proj = None  # type: Optional[nn.Linear]
        self.v_proj_local = None  # type: Optional[nn.Linear]
        self.global_stat_gate_proj = None  # type: Optional[nn.Linear]
        self.final_gate_proj = None  # type: Optional[nn.Linear]
        self.norm = None  # type: Optional[nn.LayerNorm]

    def _ensure_initialized(self, d_model: int, device: torch.device, dtype: torch.dtype):
        if not self._initialized:
            # Projections operate on the last dim (D) and are applied across (B,H,L,*)
            self.q_proj = nn.Linear(d_model, d_model).to(device=device, dtype=dtype)
            self.v_proj_local = nn.Linear(d_model, d_model).to(device=device, dtype=dtype)
            self.global_stat_gate_proj = nn.Linear(d_model, d_model).to(device=device, dtype=dtype)
            self.final_gate_proj = nn.Linear(d_model, d_model).to(device=device, dtype=dtype)
            self.norm = nn.LayerNorm(d_model).to(device=device, dtype=dtype)
            self._initialized = True

    def forward(self, *args, **kwargs):  # supports V-only or (Q,K,V)
        causal: bool = kwargs.get("causal", True)
        if len(args) == 1:
            V = args[0]
            Q = V
        elif len(args) == 3:
            Q, _K_unused, V = args
        else:
            raise TypeError("forward expects either (V,) or (Q,K,V)")
        b, h, l, d = V.shape
        self._ensure_initialized(d, V.device, V.dtype)

        assert self.q_proj is not None
        assert self.v_proj_local is not None
        assert self.global_stat_gate_proj is not None
        assert self.final_gate_proj is not None
        assert self.norm is not None

        # Step 1: Context and statistics
        if causal:
            context = torch.cumsum(V, dim=-2)
            token_indices = torch.arange(1, l + 1, device=V.device, dtype=V.dtype).view(1, 1, l, 1)
            causal_mean = context / token_indices
            causal_stat_gate = torch.sigmoid(self.global_stat_gate_proj(causal_mean))
            causal_stat = torch.cumsum(causal_stat_gate * V, dim=-2)
        else:
            context = V.sum(dim=-2, keepdim=True).expand_as(V)
            v_mean_for_stat = V.mean(dim=-2, keepdim=True)
            global_stat_gate = torch.sigmoid(self.global_stat_gate_proj(v_mean_for_stat))
            causal_stat = (global_stat_gate * V).sum(dim=-2, keepdim=True).expand_as(V)

        # Step 2: Denominator
        local_info = self.v_proj_local(V)
        denominator = torch.abs(causal_stat) + torch.abs(local_info) + 1e-8

        # Step 3: Dynamic normalization of Q and gate
        q_projected = self.q_proj(Q)
        q_normalized = q_projected / denominator
        gate = F.silu(q_normalized) * V

        # Step 4: Apply gate and final output
        final_gate = torch.sigmoid(self.final_gate_proj(gate))
        # Ensure masked tokens (where V[t]==0) produce exactly zero output
        alive = (V.abs().sum(dim=-1, keepdim=True) > 0).to(V.dtype)
        output = final_gate * context * alive
        return self.norm(output)


# --- Triton causal kernels ---
if _TRITON_AVAILABLE:
    # Optional autotuner for launch params (disabled by default; enable with EVO_AUTOTUNE=1)
    _EVO_AUTOTUNE = os.getenv("EVO_AUTOTUNE", "0") == "1"

    @lru_cache(maxsize=256)
    def _tune_prefix_params(l: int, d: int) -> tuple[int, int, int]:
        # Candidates; keep small to reduce tuning cost
        block_ms = [32, 64, 128] if l >= 512 else [32, 64]
        warps = [4, 8] if d >= 64 else [4]
        stages = [2, 3] if l >= 4096 else [2]
        device = torch.device("cuda")
        x = torch.randn(1, 1, l, d, device=device, dtype=torch.float32)
        out = torch.empty_like(x)
        best = (block_ms[0], warps[0], stages[0])
        best_t = float("inf")
        grid = (1, 1)
        torch.cuda.synchronize()
        for bm in block_ms:
            for nw in warps:
                for ns in stages:
                    # Warmup
                    for _ in range(3):
                        prefix_cumsum_fwd_kernel[grid](
                            x, out,
                            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
                            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
                            l, d,
                            BLOCK_SIZE_M=bm, BLOCK_SIZE_D=d, CAUSAL=True, ACCUM_FP64=False,
                            num_warps=nw, num_stages=ns,
                        )
                    torch.cuda.synchronize()
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(10):
                        prefix_cumsum_fwd_kernel[grid](
                            x, out,
                            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
                            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
                            l, d,
                            BLOCK_SIZE_M=bm, BLOCK_SIZE_D=d, CAUSAL=True, ACCUM_FP64=False,
                            num_warps=nw, num_stages=ns,
                        )
                    end.record(); end.synchronize()
                    t_ms = start.elapsed_time(end) / 10.0
                    if t_ms < best_t:
                        best_t = t_ms
                        best = (bm, nw, ns)
        return best

# Backend preference (auto | triton | torch)
_EVO_BACKEND = os.getenv("EVO_BACKEND", "auto").lower()
_WARNED_BACKEND_FALLBACK = False
if _TRITON_AVAILABLE:
    @triton.jit
    def prefix_cumsum_fwd_kernel(
        V, Context,
        stride_vb, stride_vh, stride_vm, stride_vk,
        stride_ob, stride_oh, stride_om, stride_ok,
        seq_len, head_dim,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_D: tl.constexpr,
        CAUSAL: tl.constexpr,
        ACCUM_FP64: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_h = tl.program_id(1)

        v_ptr = V + pid_b * stride_vb + pid_h * stride_vh
        out_ptr = Context + pid_b * stride_ob + pid_h * stride_oh

        offsets_d = tl.arange(0, BLOCK_SIZE_D)

        acc_t = tl.float64 if ACCUM_FP64 else tl.float32
        if CAUSAL:
            carry = tl.zeros([BLOCK_SIZE_D], dtype=acc_t)
            for start_m in range(0, seq_len, BLOCK_SIZE_M):
                offsets_m = start_m + tl.arange(0, BLOCK_SIZE_M)
                mask_m = offsets_m < seq_len

                v = tl.load(
                    v_ptr + offsets_m[:, None] * stride_vm + offsets_d[None, :] * stride_vk,
                    mask=mask_m[:, None], other=0.0,
                ).to(acc_t)

                local_cumsum = tl.cumsum(v, axis=0)
                c_block = carry[None, :] + local_cumsum
                tl.store(
                    out_ptr + offsets_m[:, None] * stride_om + offsets_d[None, :] * stride_ok,
                    c_block.to(Context.dtype.element_ty), mask=mask_m[:, None]
                )
                # Update carry using last valid row of local_cumsum to preserve sequential numerics
                valid_count = tl.sum(mask_m.to(tl.int32), axis=0)
                selector = (tl.arange(0, BLOCK_SIZE_M) == (valid_count - 1))
                last_seq = tl.sum(local_cumsum * selector[:, None], axis=0)
                carry = last_seq + carry
        else:
            # Compute global sum c_total then broadcast to every row
            c_total = tl.zeros([BLOCK_SIZE_D], dtype=acc_t)
            for start_m in range(0, seq_len, BLOCK_SIZE_M):
                offsets_m = start_m + tl.arange(0, BLOCK_SIZE_M)
                mask_m = offsets_m < seq_len
                v = tl.load(
                    v_ptr + offsets_m[:, None] * stride_vm + offsets_d[None, :] * stride_vk,
                    mask=mask_m[:, None], other=0.0,
                ).to(acc_t)
                c_total += tl.sum(v, axis=0)
            for start_m in range(0, seq_len, BLOCK_SIZE_M):
                offsets_m = start_m + tl.arange(0, BLOCK_SIZE_M)
                mask_m = offsets_m < seq_len
                tl.store(
                    out_ptr + offsets_m[:, None] * stride_om + offsets_d[None, :] * stride_ok,
                    c_total[None, :].to(Context.dtype.element_ty), mask=mask_m[:, None]
                )
    
    @triton.jit
    def prefix_cumsum_bwd_kernel(
        dContext, dV,
        stride_dcb, stride_dch, stride_dcm, stride_dck,
        stride_dvb, stride_dvh, stride_dvm, stride_dvk,
        seq_len, head_dim,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_D: tl.constexpr,
        CAUSAL: tl.constexpr,
        ACCUM_FP64: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_h = tl.program_id(1)

        dc_ptr = dContext + pid_b * stride_dcb + pid_h * stride_dch
        dv_ptr = dV + pid_b * stride_dvb + pid_h * stride_dvh

        offsets_d = tl.arange(0, BLOCK_SIZE_D)

        acc_t = tl.float64 if ACCUM_FP64 else tl.float32
        if CAUSAL:
            carry = tl.zeros([BLOCK_SIZE_D], dtype=acc_t)
            aligned_end = ((seq_len + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M) * BLOCK_SIZE_M
            for start_m in range(aligned_end - BLOCK_SIZE_M, -BLOCK_SIZE_M, -BLOCK_SIZE_M):
                offsets_m = start_m + tl.arange(0, BLOCK_SIZE_M)
                mask_m = (offsets_m >= 0) & (offsets_m < seq_len)
                dc = tl.load(
                    dc_ptr + offsets_m[:, None] * stride_dcm + offsets_d[None, :] * stride_dck,
                    mask=mask_m[:, None], other=0.0,
                ).to(acc_t)
                # suffix within block using forward cumsum trick
                dc = tl.where(mask_m[:, None], dc, 0.0)
                prefix_dc = tl.cumsum(dc, axis=0)
                # pick last valid row sequentially to get exact suffix-sum for this block
                valid_count = tl.sum(mask_m.to(tl.int32), axis=0)
                selector = (tl.arange(0, BLOCK_SIZE_M) == (valid_count - 1))
                sum_dc = tl.sum(dc * selector[:, None], axis=0)
                dx_block = (sum_dc[None, :] - prefix_dc + dc) + carry[None, :]
                tl.store(
                    dv_ptr + offsets_m[:, None] * stride_dvm + offsets_d[None, :] * stride_dvk,
                    dx_block.to(dV.dtype.element_ty), mask=mask_m[:, None]
                )
                carry += sum_dc
        else:
            # dx[t] = sum_j dc[j]; compute once and broadcast
            sum_dc_global = tl.zeros([BLOCK_SIZE_D], dtype=acc_t)
            for start_m in range(0, seq_len, BLOCK_SIZE_M):
                offsets_m = start_m + tl.arange(0, BLOCK_SIZE_M)
                mask_m = offsets_m < seq_len
                dc = tl.load(
                    dc_ptr + offsets_m[:, None] * stride_dcm + offsets_d[None, :] * stride_dck,
                    mask=mask_m[:, None], other=0.0,
                ).to(acc_t)
                sum_dc_global += tl.sum(dc, axis=0)
            for start_m in range(0, seq_len, BLOCK_SIZE_M):
                offsets_m = start_m + tl.arange(0, BLOCK_SIZE_M)
                mask_m = offsets_m < seq_len
                tl.store(
                    dv_ptr + offsets_m[:, None] * stride_dvm + offsets_d[None, :] * stride_dvk,
                    sum_dc_global[None, :].to(dV.dtype.element_ty), mask=mask_m[:, None]
                )

    @triton.jit
    def v6_fwd_kernel(
        Q, K, V, Out,
        Wq, bq, Wvl, bvl, Wgg, bgg, Wfg, bfg, LNw, LNb,
        stride_qb, stride_qh, stride_qm, stride_qk,
        stride_kb, stride_kh, stride_km, stride_kk,
        stride_vb, stride_vh, stride_vm, stride_vk,
        stride_ob, stride_oh, stride_om, stride_ok,
        stride_wd0, stride_wd1,  # shared for all square weight matrices (D x D)
        stride_bd,               # shared for all bias vectors (D)
        seq_len, head_dim,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_D: tl.constexpr,
        CAUSAL: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_h = tl.program_id(1)

        q_ptr = Q + pid_b * stride_qb + pid_h * stride_qh
        k_ptr = K + pid_b * stride_kb + pid_h * stride_kh  # unused logically
        v_ptr = V + pid_b * stride_vb + pid_h * stride_vh
        out_ptr = Out + pid_b * stride_ob + pid_h * stride_oh

        offsets_d = tl.arange(0, BLOCK_SIZE_D)

        # Load weights once per program
        # Shapes: (D,D) and (D)
        # We assume BLOCK_SIZE_D == head_dim
        Wq_mat = tl.load(Wq + offsets_d[:, None] * stride_wd0 + offsets_d[None, :] * stride_wd1)
        bq_vec = tl.load(bq + offsets_d * stride_bd)
        Wvl_mat = tl.load(Wvl + offsets_d[:, None] * stride_wd0 + offsets_d[None, :] * stride_wd1)
        bvl_vec = tl.load(bvl + offsets_d * stride_bd)
        Wgg_mat = tl.load(Wgg + offsets_d[:, None] * stride_wd0 + offsets_d[None, :] * stride_wd1)
        bgg_vec = tl.load(bgg + offsets_d * stride_bd)
        Wfg_mat = tl.load(Wfg + offsets_d[:, None] * stride_wd0 + offsets_d[None, :] * stride_wd1)
        bfg_vec = tl.load(bfg + offsets_d * stride_bd)
        ln_w = tl.load(LNw + offsets_d * stride_bd)
        ln_b = tl.load(LNb + offsets_d * stride_bd)

        if CAUSAL:
            carry_ctx = tl.zeros([BLOCK_SIZE_D], dtype=tl.float32)
            carry_stat = tl.zeros([BLOCK_SIZE_D], dtype=tl.float32)
            for start_m in range(0, seq_len, BLOCK_SIZE_M):
                offsets_m = start_m + tl.arange(0, BLOCK_SIZE_M)
                mask_m = offsets_m < seq_len

                q = tl.load(q_ptr + offsets_m[:, None] * stride_qm + offsets_d[None, :] * stride_qk, mask=mask_m[:, None], other=0.0).to(tl.float32)
                v = tl.load(v_ptr + offsets_m[:, None] * stride_vm + offsets_d[None, :] * stride_vk, mask=mask_m[:, None], other=0.0).to(tl.float32)

                # context cumsum(V)
                local_cumsum_v = tl.cumsum(v, axis=0)
                c_block = carry_ctx[None, :] + local_cumsum_v
                carry_ctx += tl.sum(v, axis=0)

                # q_proj = q @ Wq^T + bq  => sum over last dim with broadcast
                q_proj = tl.sum(q[:, None, :] * Wq_mat[None, :, :], axis=2) + bq_vec[None, :]

                # local_info = v @ Wvl^T + bvl
                local_info = tl.sum(v[:, None, :] * Wvl_mat[None, :, :], axis=2) + bvl_vec[None, :]

                # causal_mean = c_block / (t)
                t_idx = (start_m + 1) + tl.arange(0, BLOCK_SIZE_M)
                t_idx = tl.where(mask_m, t_idx, 1)
                denom_t = t_idx[:, None].to(tl.float32)
                causal_mean = c_block / denom_t

                # gate for stat: sigmoid(Wgg @ causal_mean + bgg)
                gstat_lin = tl.sum(causal_mean[:, None, :] * Wgg_mat[None, :, :], axis=2) + bgg_vec[None, :]
                gstat = 1.0 / (1.0 + tl.exp(-gstat_lin))
                v_gated = gstat * v

                local_cumsum_vg = tl.cumsum(v_gated, axis=0)
                stat_block = carry_stat[None, :] + local_cumsum_vg
                carry_stat += tl.sum(v_gated, axis=0)

                denominator = tl.abs(stat_block) + tl.abs(local_info) + 1e-8
                q_norm = q_proj / denominator

                # gate = silu(q_norm) * v
                sig = 1.0 / (1.0 + tl.exp(-q_norm))
                silu = q_norm * sig
                gate = silu * v

                # final gate
                f_lin = tl.sum(gate[:, None, :] * Wfg_mat[None, :, :], axis=2) + bfg_vec[None, :]
                f_gate = 1.0 / (1.0 + tl.exp(-f_lin))

                o0 = f_gate * c_block

                # LayerNorm over last dim
                mean = tl.sum(o0, axis=1) / head_dim
                xmu = o0 - mean[:, None]
                var = tl.sum(xmu * xmu, axis=1) / head_dim
                inv = tl.rsqrt(var + 1e-5)
                normed = xmu * inv[:, None]
                out = normed * ln_w[None, :] + ln_b[None, :]

                tl.store(out_ptr + offsets_m[:, None] * stride_om + offsets_d[None, :] * stride_ok, out.to(Out.dtype.element_ty), mask=mask_m[:, None])
        else:
            # Global context sum
            c_total = tl.zeros([BLOCK_SIZE_D], dtype=tl.float32)
            for start_m in range(0, seq_len, BLOCK_SIZE_M):
                offsets_m = start_m + tl.arange(0, BLOCK_SIZE_M)
                mask_m = offsets_m < seq_len
                v = tl.load(v_ptr + offsets_m[:, None] * stride_vm + offsets_d[None, :] * stride_vk, mask=mask_m[:, None], other=0.0).to(tl.float32)
                c_total += tl.sum(v, axis=0)

            # v_mean and global stat gate
            v_mean = c_total / seq_len
            gstat_lin = tl.sum(v_mean[None, :] * Wgg_mat, axis=1) + bgg_vec
            gstat = 1.0 / (1.0 + tl.exp(-gstat_lin))
            stat_vec = gstat * c_total

            for start_m in range(0, seq_len, BLOCK_SIZE_M):
                offsets_m = start_m + tl.arange(0, BLOCK_SIZE_M)
                mask_m = offsets_m < seq_len

                q = tl.load(q_ptr + offsets_m[:, None] * stride_qm + offsets_d[None, :] * stride_qk, mask=mask_m[:, None], other=0.0).to(tl.float32)
                v = tl.load(v_ptr + offsets_m[:, None] * stride_vm + offsets_d[None, :] * stride_vk, mask=mask_m[:, None], other=0.0).to(tl.float32)

                q_proj = tl.sum(q[:, None, :] * Wq_mat[None, :, :], axis=2) + bq_vec[None, :]
                local_info = tl.sum(v[:, None, :] * Wvl_mat[None, :, :], axis=2) + bvl_vec[None, :]

                context = c_total[None, :]
                stat_block = stat_vec[None, :]
                denominator = tl.abs(stat_block) + tl.abs(local_info) + 1e-8
                q_norm = q_proj / denominator

                sig = 1.0 / (1.0 + tl.exp(-q_norm))
                silu = q_norm * sig
                gate = silu * v

                f_lin = tl.sum(gate[:, None, :] * Wfg_mat[None, :, :], axis=2) + bfg_vec[None, :]
                f_gate = 1.0 / (1.0 + tl.exp(-f_lin))
                o0 = f_gate * context

                mean = tl.sum(o0, axis=1) / head_dim
                xmu = o0 - mean[:, None]
                var = tl.sum(xmu * xmu, axis=1) / head_dim
                inv = tl.rsqrt(var + 1e-5)
                normed = xmu * inv[:, None]
                out = normed * ln_w[None, :] + ln_b[None, :]

                tl.store(out_ptr + offsets_m[:, None] * stride_om + offsets_d[None, :] * stride_ok, out.to(Out.dtype.element_ty), mask=mask_m[:, None])

if _TRITON_AVAILABLE:
    class PrefixCumsumFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, causal: bool = True, block_m: int | None = None, num_warps: int | None = None, num_stages: int | None = None, accum_fp64: bool = False):
            b, h, l, d = x.shape
            out = torch.empty_like(x)

            if x.is_cuda and _TRITON_AVAILABLE and causal:
                # Heuristic + optional autotune
                if block_m is None and _EVO_AUTOTUNE:
                    tuned_bm, tuned_warps, tuned_stages = _tune_prefix_params(l, d)
                    BLOCK_SIZE_M = tuned_bm
                    num_warps_launch = tuned_warps if num_warps is None else int(num_warps)
                    num_stages_launch = tuned_stages if num_stages is None else int(num_stages)
                else:
                    if block_m is None:
                        if l >= 16384:
                            BLOCK_SIZE_M = 256
                        elif l >= 8192:
                            BLOCK_SIZE_M = 128
                        elif l >= 2048:
                            BLOCK_SIZE_M = 64
                        else:
                            BLOCK_SIZE_M = 32
                    else:
                        BLOCK_SIZE_M = int(block_m)
                    if num_warps is None:
                        num_warps_launch = 8 if d >= 128 else 4
                    else:
                        num_warps_launch = int(num_warps)
                    num_stages_launch = (3 if l >= 8192 else 2) if num_stages is None else int(num_stages)
                BLOCK_SIZE_D = d
                grid = (b, h)

                prefix_cumsum_fwd_kernel[grid](
                    x, out,
                    x.stride(0), x.stride(1), x.stride(2), x.stride(3),
                    out.stride(0), out.stride(1), out.stride(2), out.stride(3),
                    l, d,
                    BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_D=BLOCK_SIZE_D, CAUSAL=causal,
                    ACCUM_FP64=accum_fp64,
                    num_warps=num_warps_launch, num_stages=num_stages_launch,
                )
            else:
                if causal:
                    out = torch.cumsum(x, dim=-2)
                else:
                    # Non-causal path: cuBLAS-backed sum+expand is faster than a custom kernel
                    out = x.sum(dim=-2, keepdim=True).expand_as(x)

            ctx.seq_len = l
            ctx.block_m = (128 if l >= 8192 else 64 if l >= 2048 else 32) if block_m is None else int(block_m)
            ctx.causal = causal
            ctx.num_warps = num_warps
            ctx.num_stages = num_stages
            ctx.accum_fp64 = accum_fp64
            return out

        @staticmethod
        def backward(ctx, dOut):
            l = ctx.seq_len
            BLOCK_SIZE_M = ctx.block_m
            b, h, _, d = dOut.shape
            dV = torch.empty_like(dOut)

            if dOut.is_cuda and _TRITON_AVAILABLE and ctx.causal:
                grid = (b, h)
                # Heuristic launch params similar to forward for performance
                num_warps_launch = 8 if d >= 128 else 4
                num_stages_launch = 2
                prefix_cumsum_bwd_kernel[grid](
                    dOut, dV,
                    dOut.stride(0), dOut.stride(1), dOut.stride(2), dOut.stride(3),
                    dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3),
                    l, d,
                    BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_D=d, CAUSAL=ctx.causal,
                    ACCUM_FP64=getattr(ctx, "accum_fp64", False),
                    num_warps=num_warps_launch, num_stages=num_stages_launch,
                )
            else:
                if ctx.causal:
                    dV = torch.flip(torch.cumsum(torch.flip(dOut, dims=[-2]), dim=-2), dims=[-2])
                else:
                    # Non-causal gradient: broadcast of the global sum
                    dV = dOut.sum(dim=-2, keepdim=True).expand_as(dOut)

            return dV, None, None, None, None

    class EvoAttentionCausalFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, Q, K, V, causal: bool = True, accum_dtype: torch.dtype | None = None, block_m: int | None = None, num_warps: int | None = None, num_stages: int | None = None):
            batch, heads, seq_len, dim = V.shape
            output = torch.empty_like(V)

            # Heuristics for tuning and accumulation
            if block_m is None:
                if seq_len >= 8192:
                    BLOCK_SIZE_M = 128
                elif seq_len >= 2048:
                    BLOCK_SIZE_M = 64
                else:
                    BLOCK_SIZE_M = 32
            else:
                BLOCK_SIZE_M = int(block_m)
            BLOCK_SIZE_D = dim
            grid = (batch, heads)

            if num_warps is None:
                num_warps_launch = 8 if dim >= 128 else 4
            else:
                num_warps_launch = int(num_warps)
            if num_stages is None:
                num_stages_launch = 2
            else:
                num_stages_launch = int(num_stages)

            accum_fp64 = False
            if accum_dtype is not None:
                accum_fp64 = accum_dtype == torch.float64
            else:
                accum_fp64 = (Q.dtype == torch.float64) or (K.dtype == torch.float64) or (V.dtype == torch.float64)

            # choose prefix dtype consistent with accumulation precision
            cp_dtype = torch.float64 if accum_fp64 else torch.float32
            if causal:
                num_blocks = (seq_len + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
                carry_prefix = torch.empty((batch, heads, num_blocks, dim), device=Q.device, dtype=cp_dtype)
            else:
                # minimal placeholder, not used in non-causal path
                carry_prefix = torch.empty((batch, heads, 1, dim), device=Q.device, dtype=cp_dtype)

            evo_causal_fwd_kernel[grid](
                Q, K, V, output, carry_prefix,
                Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
                K.stride(0), K.stride(1), K.stride(2), K.stride(3),
                V.stride(0), V.stride(1), V.stride(2), V.stride(3),
                output.stride(0), output.stride(1), output.stride(2), output.stride(3),
                carry_prefix.stride(0), carry_prefix.stride(1), carry_prefix.stride(2), carry_prefix.stride(3),
                seq_len, dim,
                BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_D=BLOCK_SIZE_D, CAUSAL=causal, ACCUM_FP64=accum_fp64,
                num_warps=num_warps_launch, num_stages=num_stages_launch,
            )

            ctx.save_for_backward(Q, K, V, carry_prefix)
            ctx.seq_len = seq_len
            ctx.dim = dim
            ctx.block_size_m = BLOCK_SIZE_M
            ctx.causal = causal
            ctx.accum_fp64 = accum_fp64
            return output

        @staticmethod
        def backward(ctx, dOut):
            Q, K, V, carry_prefix = ctx.saved_tensors
            batch, heads, seq_len, dim = V.shape

            dQ = torch.empty_like(Q)
            dK = torch.empty_like(K)
            dV = torch.empty_like(V)

            BLOCK_SIZE_M = ctx.block_size_m
            BLOCK_SIZE_D = dim
            grid = (batch, heads)

            evo_causal_bwd_kernel[grid](
                Q, K, V, dOut, carry_prefix,
                dQ, dK, dV,
                Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
                K.stride(0), K.stride(1), K.stride(2), K.stride(3),
                V.stride(0), V.stride(1), V.stride(2), V.stride(3),
                dOut.stride(0), dOut.stride(1), dOut.stride(2), dOut.stride(3),
                dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3),
                dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3),
                dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3),
                carry_prefix.stride(0), carry_prefix.stride(1), carry_prefix.stride(2), carry_prefix.stride(3),
                seq_len, dim,
                BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_D=BLOCK_SIZE_D, CAUSAL=ctx.causal, ACCUM_FP64=ctx.accum_fp64,
            )

            # Return gradients for all forward inputs: (Q, K, V, causal, accum_dtype, block_m, num_warps, num_stages)
            return dQ, dK, dV, None, None, None, None, None


    class EvoAttentionCausalTriton(nn.Module):
        """EvoAttention core with the same parameters as the Torch reference.

        Hybrid path: Triton kernels compute prefix cumsums (causal or global), all other
        elementwise/linear operations are computed via PyTorch for correctness and speed.
        """

        def __init__(self):
            super().__init__()
            self._initialized = False
            self.q_proj = None  # type: Optional[nn.Linear]
            self.v_proj_local = None  # type: Optional[nn.Linear]
            self.global_stat_gate_proj = None  # type: Optional[nn.Linear]
            self.final_gate_proj = None  # type: Optional[nn.Linear]
            self.norm = None  # type: Optional[nn.LayerNorm]

        def _ensure_initialized(self, d_model: int, device: torch.device, dtype: torch.dtype):
            if not self._initialized:
                self.q_proj = nn.Linear(d_model, d_model).to(device=device, dtype=dtype)
                self.v_proj_local = nn.Linear(d_model, d_model).to(device=device, dtype=dtype)
                self.global_stat_gate_proj = nn.Linear(d_model, d_model).to(device=device, dtype=dtype)
                self.final_gate_proj = nn.Linear(d_model, d_model).to(device=device, dtype=dtype)
                self.norm = nn.LayerNorm(d_model).to(device=device, dtype=dtype)
                self._initialized = True

        def sync_from(self, ref: "EvoAttentionCausalTorch") -> None:
            # Copy parameters from torch reference into this module (device/dtype preserved)
            assert ref.norm is not None and ref.q_proj is not None and ref.v_proj_local is not None and ref.global_stat_gate_proj is not None and ref.final_gate_proj is not None
            device = ref.norm.weight.device
            dtype = ref.norm.weight.dtype
            d_model = ref.norm.normalized_shape[0]
            self._ensure_initialized(d_model, device, dtype)
            assert self.q_proj is not None and self.v_proj_local is not None and self.global_stat_gate_proj is not None and self.final_gate_proj is not None and self.norm is not None
            with torch.no_grad():
                self.q_proj.weight.copy_(ref.q_proj.weight.to(device=device, dtype=dtype))
                self.q_proj.bias.copy_(ref.q_proj.bias.to(device=device, dtype=dtype))
                self.v_proj_local.weight.copy_(ref.v_proj_local.weight.to(device=device, dtype=dtype))
                self.v_proj_local.bias.copy_(ref.v_proj_local.bias.to(device=device, dtype=dtype))
                self.global_stat_gate_proj.weight.copy_(ref.global_stat_gate_proj.weight.to(device=device, dtype=dtype))
                self.global_stat_gate_proj.bias.copy_(ref.global_stat_gate_proj.bias.to(device=device, dtype=dtype))
                self.final_gate_proj.weight.copy_(ref.final_gate_proj.weight.to(device=device, dtype=dtype))
                self.final_gate_proj.bias.copy_(ref.final_gate_proj.bias.to(device=device, dtype=dtype))
                self.norm.weight.copy_(ref.norm.weight.to(device=device, dtype=dtype))
                self.norm.bias.copy_(ref.norm.bias.to(device=device, dtype=dtype))

        def forward(self, V_only_or_Q=None, K=None, V=None, *, causal: bool = True, accum_dtype: torch.dtype | None = None, block_m: int | None = None, num_warps: int | None = None, num_stages: int | None = None):
            # Support V-only call or (Q,K,V)
            if V is None and K is None:
                V = V_only_or_Q
                Q = V
            else:
                Q = V_only_or_Q
            b, h, l, d = V.shape
            self._ensure_initialized(d, V.device, V.dtype)

            assert self.q_proj is not None
            assert self.v_proj_local is not None
            assert self.global_stat_gate_proj is not None
            assert self.final_gate_proj is not None
            assert self.norm is not None

            # Heuristic launch params
            if block_m is None:
                if l >= 8192:
                    BLOCK_SIZE_M = 128
                elif l >= 2048:
                    BLOCK_SIZE_M = 64
                else:
                    BLOCK_SIZE_M = 32
            else:
                BLOCK_SIZE_M = int(block_m)
            BLOCK_SIZE_D = d
            grid = (b, h)

            # 1) Context and 2) Stat
            if causal:
                context = PrefixCumsumFunction.apply(V, True, BLOCK_SIZE_M, num_warps, num_stages)
                token_indices = torch.arange(1, l + 1, device=V.device, dtype=V.dtype).view(1, 1, l, 1)
                causal_mean = context / token_indices
                gstat = torch.sigmoid(self.global_stat_gate_proj(causal_mean))
                Vg = gstat * V
                causal_stat = PrefixCumsumFunction.apply(Vg, True, BLOCK_SIZE_M, num_warps, num_stages)
            else:
                # Pure PyTorch path (faster; avoids custom autograd overhead)
                context = V.sum(dim=-2, keepdim=True).expand_as(V)
                v_mean_for_stat = V.mean(dim=-2, keepdim=True)
                global_stat_gate = torch.sigmoid(self.global_stat_gate_proj(v_mean_for_stat))
                causal_stat = (global_stat_gate * V).sum(dim=-2, keepdim=True).expand_as(V)

            # 3) Elementwise via PyTorch (fast)
            local_info = self.v_proj_local(V)
            denominator = torch.abs(causal_stat) + torch.abs(local_info) + 1e-8
            q_projected = self.q_proj(Q)
            q_normalized = q_projected / denominator
            gate = F.silu(q_normalized) * V
            final_gate = torch.sigmoid(self.final_gate_proj(gate))
            output = final_gate * context

            alive = (V.abs().sum(dim=-1, keepdim=True) > 0).to(V.dtype)
            return self.norm(output * alive)


# ------------------ Tests ------------------
# Set this flag to True once the Triton backward kernel is implemented.
TRITON_V6_BACKWARD_READY = True
def test_forward_correctness_small():
    if not torch.cuda.is_available():
        print("CUDA not available, skipping causal forward correctness test.")
        return

    device = "cuda"
    dtype = torch.float32
    torch.manual_seed(0)

    B, H, L, D = 2, 4, 1024, 64
    Q = torch.randn(B, H, L, D, device=device, dtype=dtype)
    K = torch.randn(B, H, L, D, device=device, dtype=dtype)
    V = torch.randn(B, H, L, D, device=device, dtype=dtype)

    ref = EvoAttentionCausalTorch().to(device)
    trt = EvoAttentionCausalTriton().to(device)

    # initialize ref modules and synchronize parameters to ensure identical computation
    with torch.no_grad():
        _ = ref(Q, K, V)
    if hasattr(trt, "sync_from"):
        trt.sync_from(ref)

    with torch.no_grad():
        out_ref = ref(Q, K, V)
        out_trt = trt(Q, K, V)

    diff = (out_ref - out_trt).abs().max().item()
    print(f"Causal forward correctness (max abs diff): {diff:.2e}")
    assert diff < 5e-4, "Causal forward mismatch"


def test_backward_correctness_small():
    if not torch.cuda.is_available():
        print("CUDA not available, skipping causal backward correctness test.")
        return
    if not TRITON_V6_BACKWARD_READY:
        print("Triton V6 backward not implemented yet, skipping causal backward correctness test.")
        return

    device = "cuda"
    dtype = torch.float32
    torch.manual_seed(0)

    B, H, L, D = 1, 2, 512, 64
    Q1 = torch.randn(B, H, L, D, device=device, dtype=dtype, requires_grad=True)
    K1 = torch.randn(B, H, L, D, device=device, dtype=dtype, requires_grad=True)
    V1 = torch.randn(B, H, L, D, device=device, dtype=dtype, requires_grad=True)

    Q2 = Q1.clone().detach().requires_grad_(True)
    K2 = K1.clone().detach().requires_grad_(True)
    V2 = V1.clone().detach().requires_grad_(True)

    trt = EvoAttentionCausalTriton().to(device)
    ref = EvoAttentionCausalTorch().to(device)

    out_tr = trt(Q1, K1, V1)
    loss_tr = out_tr.sum()
    loss_tr.backward()

    out_rf = ref(Q2, K2, V2)
    loss_rf = out_rf.sum()
    loss_rf.backward()

    dq1 = Q1.grad if Q1.grad is not None else torch.zeros_like(Q1)
    dq2 = Q2.grad if Q2.grad is not None else torch.zeros_like(Q2)
    dk1 = K1.grad if K1.grad is not None else torch.zeros_like(K1)
    dk2 = K2.grad if K2.grad is not None else torch.zeros_like(K2)
    dv1 = V1.grad if V1.grad is not None else torch.zeros_like(V1)
    dv2 = V2.grad if V2.grad is not None else torch.zeros_like(V2)
    dq_diff = (dq1 - dq2).abs().max().item()
    dk_diff = (dk1 - dk2).abs().max().item()
    dv_diff = (dv1 - dv2).abs().max().item()
    print(f"Causal backward correctness max diffs: dQ={dq_diff:.2e}, dK={dk_diff:.2e}, dV={dv_diff:.2e}")
    assert dq_diff < 5e-4 and dk_diff < 5e-4 and dv_diff < 5e-4, "Causal backward mismatch"


def test_forward_correctness_small_noncausal():
    if not torch.cuda.is_available():
        print("CUDA not available, skipping non-causal forward correctness test.")
        return

    device = "cuda"
    dtype = torch.float32
    torch.manual_seed(0)

    B, H, L, D = 2, 4, 1024, 64
    Q = torch.randn(B, H, L, D, device=device, dtype=dtype)
    K = torch.randn(B, H, L, D, device=device, dtype=dtype)
    V = torch.randn(B, H, L, D, device=device, dtype=dtype)

    ref = EvoAttentionCausalTorch().to(device)
    trt = EvoAttentionCausalTriton().to(device)
    with torch.no_grad():
        _ = ref(Q, K, V, causal=False)
    if hasattr(trt, "sync_from"):
        trt.sync_from(ref)

    with torch.no_grad():
        out_ref = ref(Q, K, V, causal=False)
        out_trt = trt(Q, K, V, causal=False)

    diff = (out_ref - out_trt).abs().max().item()
    print(f"Non-causal forward correctness (max abs diff): {diff:.2e}")
    assert diff < 5e-4, "Non-causal forward mismatch"


def test_backward_correctness_small_noncausal():
    if not torch.cuda.is_available():
        print("CUDA not available, skipping non-causal backward correctness test.")
    if not TRITON_V6_BACKWARD_READY:
        print("Triton V6 backward not implemented yet, skipping non-causal backward correctness test.")
        return

    device = "cuda"
    dtype = torch.float32
    torch.manual_seed(0)

    B, H, L, D = 1, 2, 512, 64
    Q1 = torch.randn(B, H, L, D, device=device, dtype=dtype, requires_grad=True)
    K1 = torch.randn(B, H, L, D, device=device, dtype=dtype, requires_grad=True)
    V1 = torch.randn(B, H, L, D, device=device, dtype=dtype, requires_grad=True)

    Q2 = Q1.clone().detach().requires_grad_(True)
    K2 = K1.clone().detach().requires_grad_(True)
    V2 = V1.clone().detach().requires_grad_(True)

    trt = EvoAttentionCausalTriton().to(device)
    ref = EvoAttentionCausalTorch().to(device)

    out_tr = trt(Q1, K1, V1, causal=False)
    loss_tr = out_tr.sum()
    loss_tr.backward()

    out_rf = ref(Q2, K2, V2, causal=False)
    loss_rf = out_rf.sum()
    loss_rf.backward()

    dq1 = Q1.grad if Q1.grad is not None else torch.zeros_like(Q1)
    dq2 = Q2.grad if Q2.grad is not None else torch.zeros_like(Q2)
    dk1 = K1.grad if K1.grad is not None else torch.zeros_like(K1)
    dk2 = K2.grad if K2.grad is not None else torch.zeros_like(K2)
    dv1 = V1.grad if V1.grad is not None else torch.zeros_like(V1)
    dv2 = V2.grad if V2.grad is not None else torch.zeros_like(V2)
    dq_diff = (dq1 - dq2).abs().max().item()
    dk_diff = (dk1 - dk2).abs().max().item()
    dv_diff = (dv1 - dv2).abs().max().item()
    print(f"Non-causal backward correctness max diffs: dQ={dq_diff:.2e}, dK={dk_diff:.2e}, dV={dv_diff:.2e}")
    assert dq_diff < 5e-4 and dk_diff < 5e-4 and dv_diff < 5e-4, "Non-causal backward mismatch"


def test_multi_dtype_correctness():
    if not torch.cuda.is_available():
        print("CUDA not available, skipping multi-dtype correctness test.")
        return

    device = "cuda"
    torch.manual_seed(0)

    dtypes = [torch.float16, torch.bfloat16, torch.float32]
    # torch.float64 often unsupported on consumer GPUs for Triton; keep reference-only path

    B, H, L, D = 1, 2, 256, 64
    for causal in (True, False):
        for dt in dtypes:
            Q = torch.randn(B, H, L, D, device=device, dtype=dt)
            K = torch.randn(B, H, L, D, device=device, dtype=dt)
            V = torch.randn(B, H, L, D, device=device, dtype=dt)

            ref = EvoAttentionCausalTorch().to(device)
            trt = EvoAttentionCausalTriton().to(device)
            with torch.no_grad():
                _ = ref(Q, K, V, causal=causal)
            if hasattr(trt, "sync_from"):
                trt.sync_from(ref)

            with torch.no_grad():
                out_ref = ref(Q.float(), K.float(), V.float(), causal=causal).to(dt)
                out_trt = trt(Q, K, V, causal=causal)

            diff = (out_ref - out_trt).abs().max().item()
            print(f"Multi-dtype {str(dt)} causal={causal} max diff: {diff:.2e}")
            tol = 3e-3 if dt in (torch.float16, torch.bfloat16) else 5e-4
            assert diff < tol, "Multi-dtype forward mismatch"


def test_forward_perf():
    if not torch.cuda.is_available():
        print("CUDA not available, skipping causal performance test.")
        return

    device = "cuda"
    dtype = torch.float32
    B, H, L, D = 2, 2, 2048, 64

    Q = torch.randn(B, H, L, D, device=device, dtype=dtype)
    K = torch.randn(B, H, L, D, device=device, dtype=dtype)
    V = torch.randn(B, H, L, D, device=device, dtype=dtype)

    ref = EvoAttentionCausalTorch().to(device)
    trt = EvoAttentionCausalTriton().to(device)

    def bench_ms(fn, warmup=10, iters=10, repeats=10):
        # warmup
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        samples = []
        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iters):
                fn()
            end.record()
            end.synchronize()
            ms = start.elapsed_time(end) / iters
            samples.append(ms)
        mean = sum(samples) / len(samples)
        if len(samples) > 1:
            var = sum((x - mean) ** 2 for x in samples) / (len(samples) - 1)
            std = math.sqrt(var)
            ci95 = 1.96 * std / math.sqrt(len(samples))
        else:
            ci95 = 0.0
        return mean, ci95

    m_ref, ci_ref = bench_ms(lambda: ref(Q, K, V))
    m_trt, ci_trt = bench_ms(lambda: trt(Q, K, V))

    print(
        f"Causal forward time: Torch={m_ref:.2f}±{ci_ref:.2f} ms, "
        f"Triton={m_trt:.2f}±{ci_trt:.2f} ms, speedup={m_ref / m_trt:.2f}x"
    )


def test_backward_perf():
    if not torch.cuda.is_available():
        print("CUDA not available, skipping causal backward performance test.")
        return

    device = "cuda"
    dtype = torch.float32

    # Use a configuration that fits into memory for backward
    B, H, L, D = 2, 2, 2048, 64

    def make_inputs():
        torch.manual_seed(0)
        Q = torch.randn(B, H, L, D, device=device, dtype=dtype, requires_grad=True)
        K = torch.randn(B, H, L, D, device=device, dtype=dtype, requires_grad=True)
        V = torch.randn(B, H, L, D, device=device, dtype=dtype, requires_grad=True)
        return Q, K, V

    trt = EvoAttentionCausalTriton().to(device)
    ref = EvoAttentionCausalTorch().to(device)

    def bench_ms(fn, warmup=5, iters=10, repeats=10):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        samples = []
        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iters):
                fn()
            end.record()
            end.synchronize()
            ms = start.elapsed_time(end) / iters
            samples.append(ms)
        mean = sum(samples) / len(samples)
        if len(samples) > 1:
            var = sum((x - mean) ** 2 for x in samples) / (len(samples) - 1)
            std = math.sqrt(var)
            ci95 = 1.96 * std / math.sqrt(len(samples))
        else:
            ci95 = 0.0
        return mean, ci95

    # Prepare a fresh set of inputs once per benchmarked function
    def bench_backward(module):
        Q, K, V = make_inputs()
        def step():
            out = module(Q, K, V)
            loss = out.sum()
            loss.backward()
            Q.grad = None; K.grad = None; V.grad = None
        return bench_ms(step)

    m_trt, ci_trt = bench_backward(trt)
    m_ref, ci_ref = bench_backward(ref)

    print(
        f"Causal backward time: Torch={m_ref:.2f}±{ci_ref:.2f} ms, "
        f"Triton={m_trt:.2f}±{ci_trt:.2f} ms, speedup={m_ref / m_trt:.2f}x"
    )


def test_forward_perf_noncausal():
    if not torch.cuda.is_available():
        print("CUDA not available, skipping non-causal performance test.")
        return

    device = "cuda"
    dtype = torch.float32
    B, H, L, D = 4, 2, 2048, 64

    Q = torch.randn(B, H, L, D, device=device, dtype=dtype)
    K = torch.randn(B, H, L, D, device=device, dtype=dtype)
    V = torch.randn(B, H, L, D, device=device, dtype=dtype)

    ref = EvoAttentionCausalTorch().to(device)
    trt = EvoAttentionCausalTriton().to(device)

    def bench_ms(fn, warmup=10, iters=10, repeats=10):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        samples = []
        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iters):
                fn()
            end.record()
            end.synchronize()
            ms = start.elapsed_time(end) / iters
            samples.append(ms)
        mean = sum(samples) / len(samples)
        if len(samples) > 1:
            var = sum((x - mean) ** 2 for x in samples) / (len(samples) - 1)
            std = math.sqrt(var)
            ci95 = 1.96 * std / math.sqrt(len(samples))
        else:
            ci95 = 0.0
        return mean, ci95

    m_ref, ci_ref = bench_ms(lambda: ref(Q, K, V, causal=False))
    m_trt, ci_trt = bench_ms(lambda: trt(Q, K, V, causal=False))

    print(
        f"Non-causal forward time: Torch={m_ref:.2f}±{ci_ref:.2f} ms, "
        f"Triton={m_trt:.2f}±{ci_trt:.2f} ms, speedup={m_ref / m_trt:.2f}x"
    )


def test_backward_perf_noncausal():
    if not torch.cuda.is_available():
        print("CUDA not available, skipping non-causal backward performance test.")
        return

    device = "cuda"
    dtype = torch.float32

    B, H, L, D = 2, 2, 2048, 64

    def make_inputs():
        torch.manual_seed(0)
        Q = torch.randn(B, H, L, D, device=device, dtype=dtype, requires_grad=True)
        K = torch.randn(B, H, L, D, device=device, dtype=dtype, requires_grad=True)
        V = torch.randn(B, H, L, D, device=device, dtype=dtype, requires_grad=True)
        return Q, K, V

    trt = EvoAttentionCausalTriton().to(device)
    ref = EvoAttentionCausalTorch().to(device)

    def bench_ms(fn, warmup=5, iters=10, repeats=10):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        samples = []
        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iters):
                fn()
            end.record()
            end.synchronize()
            ms = start.elapsed_time(end) / iters
            samples.append(ms)
        mean = sum(samples) / len(samples)
        if len(samples) > 1:
            var = sum((x - mean) ** 2 for x in samples) / (len(samples) - 1)
            std = math.sqrt(var)
            ci95 = 1.96 * std / math.sqrt(len(samples))
        else:
            ci95 = 0.0
        return mean, ci95

    def bench_backward(module):
        Q, K, V = make_inputs()
        def step():
            out = module(Q, K, V, causal=False)
            loss = out.sum()
            loss.backward()
            Q.grad = None; K.grad = None; V.grad = None
        return bench_ms(step)

    m_trt, ci_trt = bench_backward(trt)
    m_ref, ci_ref = bench_backward(ref)

    print(
        f"Non-causal backward time: Torch={m_ref:.2f}±{ci_ref:.2f} ms, "
        f"Triton={m_trt:.2f}±{ci_trt:.2f} ms, speedup={m_ref / m_trt:.2f}x"
    )


if __name__ == "__main__":
    test_forward_correctness_small()
    test_backward_correctness_small()
    test_forward_correctness_small_noncausal()
    test_backward_correctness_small_noncausal()
    test_forward_perf()
    test_backward_perf()
    test_forward_perf_noncausal()
    test_backward_perf_noncausal()

