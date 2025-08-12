# ruff: noqa
# mypy: ignore-errors
# pyright: reportUndefinedVariable=false, reportGeneralTypeIssues=false
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except Exception:  # pragma: no cover - platform without Triton
    triton = None
    tl = None
    _TRITON_AVAILABLE = False
import math


class EvoAttentionCausalTorch(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def _rms_norm(x, eps: float = 1e-5):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)

    @staticmethod
    def _get_l2_norm(x, eps: float = 1e-8):
        return torch.norm(x, p=2, dim=-1, keepdim=True) + eps

    @staticmethod
    def _swiglu(gate, value):
        return F.silu(gate) * value

    def forward(self, Q, K, V, causal: bool = True):
        v_swiglu = F.silu(V) * V
        if causal:
            c = torch.cumsum(v_swiglu, dim=-2)
        else:
            c_total = v_swiglu.sum(dim=-2, keepdim=True)
            c = c_total.expand_as(V)

        c_n = self._rms_norm(c)
        r = self._get_l2_norm(Q) + self._get_l2_norm(K) + 1.0
        mstate = c_n / r
        out0 = V + self._swiglu(mstate, V)
        return self._rms_norm(out0)


# --- Triton causal kernels ---
if _TRITON_AVAILABLE:
    @triton.jit
    def evo_causal_fwd_kernel(
    Q, K, V, Out, CarryPrefix,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_km, stride_kk,
    stride_vb, stride_vh, stride_vm, stride_vk,
    stride_ob, stride_oh, stride_om, stride_ok,
    stride_cpb, stride_cph, stride_cpm, stride_cpd,
    seq_len, head_dim,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    CAUSAL: tl.constexpr,
    ACCUM_FP64: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_h = tl.program_id(1)

    q_ptr = Q + pid_b * stride_qb + pid_h * stride_qh
    k_ptr = K + pid_b * stride_kb + pid_h * stride_kh
    v_ptr = V + pid_b * stride_vb + pid_h * stride_vh
    out_ptr = Out + pid_b * stride_ob + pid_h * stride_oh
    cp_base_ptr = CarryPrefix + pid_b * stride_cpb + pid_h * stride_cph

    offsets_d = tl.arange(0, BLOCK_SIZE_D)
    DTYPE = tl.float64 if ACCUM_FP64 else tl.float32
    if CAUSAL:
        carry_sum = tl.zeros([BLOCK_SIZE_D], dtype=DTYPE)
        carry_comp = tl.zeros([BLOCK_SIZE_D], dtype=DTYPE)

        block_idx = 0
        for start_m in range(0, seq_len, BLOCK_SIZE_M):
            offsets_m = start_m + tl.arange(0, BLOCK_SIZE_M)
            mask_m = offsets_m < seq_len

            # store prefix before processing this block
            cp_ptr = cp_base_ptr + block_idx * stride_cpm
            tl.store(cp_ptr + offsets_d * stride_cpd, carry_sum.to(CarryPrefix.dtype.element_ty))

            q = tl.load(q_ptr + offsets_m[:, None] * stride_qm + offsets_d[None, :] * stride_qk, mask=mask_m[:, None], other=0.0).to(DTYPE)
            k = tl.load(k_ptr + offsets_m[:, None] * stride_km + offsets_d[None, :] * stride_kk, mask=mask_m[:, None], other=0.0).to(DTYPE)
            v = tl.load(v_ptr + offsets_m[:, None] * stride_vm + offsets_d[None, :] * stride_vk, mask=mask_m[:, None], other=0.0).to(DTYPE)

            sig_v = tl.sigmoid(v)
            v_silu = v * sig_v
            v_swiglu = v_silu * v

            local_cumsum = tl.cumsum(v_swiglu, axis=0)
            compensated_local = local_cumsum - carry_comp[None, :]
            temp_sum = carry_sum[None, :] + compensated_local
            new_comp = (temp_sum - carry_sum[None, :]) - compensated_local

            c_block = temp_sum
            carry_sum += tl.sum(v_swiglu, axis=0)
            carry_comp += tl.sum(new_comp, axis=0)

            # RMSNorm(c)
            c_var = tl.sum(c_block * c_block, axis=1) / head_dim
            inv_rms_c = tl.rsqrt(c_var + 1e-5)
            c_n = c_block * inv_rms_c[:, None]

            # r = ||Q|| + ||K|| + 1
            l2_q = tl.sqrt(tl.sum(q * q, axis=1) + 1e-8)
            l2_k = tl.sqrt(tl.sum(k * k, axis=1) + 1e-8)
            r = l2_q + l2_k + 1.0

            mstate = c_n / r[:, None]
            g = mstate * tl.sigmoid(mstate)
            o0 = v + g * v

            out_var = tl.sum(o0 * o0, axis=1) / head_dim
            inv_rms_out = tl.rsqrt(out_var + 1e-5)
            out = o0 * inv_rms_out[:, None]

            tl.store(out_ptr + offsets_m[:, None] * stride_om + offsets_d[None, :] * stride_ok, out.to(Out.dtype.element_ty), mask=mask_m[:, None])

            block_idx += 1
    else:
        # Pass 1: accumulate global C_total = sum_t swiglu(V[t])
        global_sum = tl.zeros([BLOCK_SIZE_D], dtype=DTYPE)
        for start_m in range(0, seq_len, BLOCK_SIZE_M):
            offsets_m = start_m + tl.arange(0, BLOCK_SIZE_M)
            mask_m = offsets_m < seq_len
            v = tl.load(v_ptr + offsets_m[:, None] * stride_vm + offsets_d[None, :] * stride_vk, mask=mask_m[:, None], other=0.0).to(DTYPE)
            sig_v = tl.sigmoid(v)
            v_silu = v * sig_v
            v_swiglu = v_silu * v
            global_sum += tl.sum(v_swiglu, axis=0)

        # Normalize global c
        c_var_g = tl.sum(global_sum * global_sum) / head_dim
        inv_rms_c_g = tl.rsqrt(c_var_g + 1e-5)
        c_n_vec = global_sum * inv_rms_c_g

        # Pass 2: compute outputs using the same c for all rows
        for start_m in range(0, seq_len, BLOCK_SIZE_M):
            offsets_m = start_m + tl.arange(0, BLOCK_SIZE_M)
            mask_m = offsets_m < seq_len
            q = tl.load(q_ptr + offsets_m[:, None] * stride_qm + offsets_d[None, :] * stride_qk, mask=mask_m[:, None], other=0.0).to(DTYPE)
            k = tl.load(k_ptr + offsets_m[:, None] * stride_km + offsets_d[None, :] * stride_kk, mask=mask_m[:, None], other=0.0).to(DTYPE)
            v = tl.load(v_ptr + offsets_m[:, None] * stride_vm + offsets_d[None, :] * stride_vk, mask=mask_m[:, None], other=0.0).to(DTYPE)

            l2_q = tl.sqrt(tl.sum(q * q, axis=1) + 1e-8)
            l2_k = tl.sqrt(tl.sum(k * k, axis=1) + 1e-8)
            r = l2_q + l2_k + 1.0

            mstate = c_n_vec[None, :] / r[:, None]
            g = mstate * tl.sigmoid(mstate)
            o0 = v + g * v

            out_var = tl.sum(o0 * o0, axis=1) / head_dim
            inv_rms_out = tl.rsqrt(out_var + 1e-5)
            out = o0 * inv_rms_out[:, None]

            tl.store(out_ptr + offsets_m[:, None] * stride_om + offsets_d[None, :] * stride_ok, out.to(Out.dtype.element_ty), mask=mask_m[:, None])


    @triton.jit
    def evo_causal_bwd_kernel(
    Q, K, V, dOut, CarryPrefix,
    dQ, dK, dV,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_km, stride_kk,
    stride_vb, stride_vh, stride_vm, stride_vk,
    stride_dyb, stride_dyh, stride_dym, stride_dyk,
    stride_dqb, stride_dqh, stride_dqm, stride_dqk,
    stride_dkb, stride_dkh, stride_dkm, stride_dkk,
    stride_dvb, stride_dvh, stride_dvm, stride_dvk,
    stride_cpb, stride_cph, stride_cpm, stride_cpd,
    seq_len, head_dim,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    CAUSAL: tl.constexpr,
    ACCUM_FP64: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_h = tl.program_id(1)

    q_ptr = Q + pid_b * stride_qb + pid_h * stride_qh
    k_ptr = K + pid_b * stride_kb + pid_h * stride_kh
    v_ptr = V + pid_b * stride_vb + pid_h * stride_vh
    dy_ptr = dOut + pid_b * stride_dyb + pid_h * stride_dyh

    dq_ptr = dQ + pid_b * stride_dqb + pid_h * stride_dqh
    dk_ptr = dK + pid_b * stride_dkb + pid_h * stride_dkh
    dv_ptr = dV + pid_b * stride_dvb + pid_h * stride_dvh

    cp_base_ptr = CarryPrefix + pid_b * stride_cpb + pid_h * stride_cph

    offsets_d = tl.arange(0, BLOCK_SIZE_D)
    DTYPE = tl.float64 if ACCUM_FP64 else tl.float32
    if CAUSAL:
        carry_grad_c = tl.zeros([BLOCK_SIZE_D], dtype=DTYPE)

        aligned_end = ((seq_len + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M) * BLOCK_SIZE_M
        for start_m in range(aligned_end - BLOCK_SIZE_M, -BLOCK_SIZE_M, -BLOCK_SIZE_M):
            offsets_m = start_m + tl.arange(0, BLOCK_SIZE_M)
            mask_m = (offsets_m >= 0) & (offsets_m < seq_len)

            q = tl.load(q_ptr + offsets_m[:, None] * stride_qm + offsets_d[None, :] * stride_qk, mask=mask_m[:, None], other=0.0).to(DTYPE)
            k = tl.load(k_ptr + offsets_m[:, None] * stride_km + offsets_d[None, :] * stride_kk, mask=mask_m[:, None], other=0.0).to(DTYPE)
            v = tl.load(v_ptr + offsets_m[:, None] * stride_vm + offsets_d[None, :] * stride_vk, mask=mask_m[:, None], other=0.0).to(DTYPE)
            dy = tl.load(dy_ptr + offsets_m[:, None] * stride_dym + offsets_d[None, :] * stride_dyk, mask=mask_m[:, None], other=0.0).to(DTYPE)

            block_index = start_m // BLOCK_SIZE_M
            cp_ptr = cp_base_ptr + block_index * stride_cpm
            prefix = tl.load(cp_ptr + offsets_d * stride_cpd).to(DTYPE)

            sig_v = tl.sigmoid(v)
            v_silu = v * sig_v
            v_swiglu = v_silu * v
            local_cumsum = tl.cumsum(v_swiglu, axis=0)
            c_block = local_cumsum + prefix[None, :]

            c_var = tl.sum(c_block * c_block, axis=1) / head_dim
            inv_rms_c = tl.rsqrt(c_var + 1e-5)
            c_n = c_block * inv_rms_c[:, None]

            l2_q = tl.sqrt(tl.sum(q * q, axis=1) + 1e-8)
            l2_k = tl.sqrt(tl.sum(k * k, axis=1) + 1e-8)
            r = l2_q + l2_k + 1.0

            mstate = c_n / r[:, None]
            sig_m = tl.sigmoid(mstate)
            g = mstate * sig_m
            o0 = v * (1.0 + g)

            out_var = tl.sum(o0 * o0, axis=1) / head_dim
            inv_rms_out = tl.rsqrt(out_var + 1e-5)
            out_y = o0 * inv_rms_out[:, None]

            # d out -> d o0 (RMSNorm bwd)
            mean_dy_y = tl.sum(dy * out_y, axis=1) / head_dim
            do0 = inv_rms_out[:, None] * (dy - out_y * mean_dy_y[:, None])

            # через o0 = v * (1 + g)
            dV_from_o0 = do0 * (1.0 + g)
            dg = do0 * v

            # через g = SiLU(mstate)
            dsilu_m = sig_m * (1.0 + mstate * (1.0 - sig_m))
            dmstate = dg * dsilu_m

            # mstate = c_n / r
            dc_n = dmstate / r[:, None]
            dr = - tl.sum(dmstate * c_n, axis=1) / (r * r)

            # RMSNorm(c) bwd
            mean_dcny = tl.sum(dc_n * c_n, axis=1) / head_dim
            dc = inv_rms_c[:, None] * (dc_n - c_n * mean_dcny[:, None])

            # dv_sw via vectorized suffix-sum: s[i] = sum_{j=i}^{M-1} dc[j]
            dsilu_v = sig_v * (1.0 + v * (1.0 - sig_v))
            dc = tl.where(mask_m[:, None], dc, 0.0)
            sum_dc = tl.sum(dc, axis=0)
            prefix_dc = tl.cumsum(dc, axis=0)
            dv_sw = (sum_dc[None, :] - prefix_dc + dc) + carry_grad_c[None, :]
            dv_from_vsw = dv_sw * (dsilu_v * v + v_silu)
            dV_block = dV_from_o0 + dv_from_vsw
            tl.store(dv_ptr + offsets_m[:, None] * stride_dvm + offsets_d[None, :] * stride_dvk, dV_block.to(dV.dtype.element_ty), mask=mask_m[:, None])
            carry_grad_c = carry_grad_c + sum_dc

            # d r -> dQ, dK (blockwise)
            dQ_block = tl.where(l2_q[:, None] > 0.0, dr[:, None] * (q / l2_q[:, None]), 0.0)
            dK_block = tl.where(l2_k[:, None] > 0.0, dr[:, None] * (k / l2_k[:, None]), 0.0)

            tl.store(dq_ptr + offsets_m[:, None] * stride_dqm + offsets_d[None, :] * stride_dqk, dQ_block.to(dQ.dtype.element_ty), mask=mask_m[:, None])
            tl.store(dk_ptr + offsets_m[:, None] * stride_dkm + offsets_d[None, :] * stride_dkk, dK_block.to(dK.dtype.element_ty), mask=mask_m[:, None])
    else:
        # Non-causal backward in 3 passes
        # Pass 1: accumulate global C_total
        global_sum = tl.zeros([BLOCK_SIZE_D], dtype=DTYPE)
        for start_m in range(0, seq_len, BLOCK_SIZE_M):
            offsets_m = start_m + tl.arange(0, BLOCK_SIZE_M)
            mask_m = offsets_m < seq_len
            v = tl.load(v_ptr + offsets_m[:, None] * stride_vm + offsets_d[None, :] * stride_vk, mask=mask_m[:, None], other=0.0).to(DTYPE)
            sig_v = tl.sigmoid(v)
            v_silu = v * sig_v
            v_swiglu = v_silu * v
            global_sum += tl.sum(v_swiglu, axis=0)

        c_var_g = tl.sum(global_sum * global_sum) / head_dim
        inv_rms_c_g = tl.rsqrt(c_var_g + 1e-5)
        c_n_vec = global_sum * inv_rms_c_g

        sum_dc_global = tl.zeros([BLOCK_SIZE_D], dtype=DTYPE)

        # Pass 2: compute do0 path, dQ/dK, partial dV_from_o0 and accumulate sum(dc)
        for start_m in range(0, seq_len, BLOCK_SIZE_M):
            offsets_m = start_m + tl.arange(0, BLOCK_SIZE_M)
            mask_m = offsets_m < seq_len

            q = tl.load(q_ptr + offsets_m[:, None] * stride_qm + offsets_d[None, :] * stride_qk, mask=mask_m[:, None], other=0.0).to(DTYPE)
            k = tl.load(k_ptr + offsets_m[:, None] * stride_km + offsets_d[None, :] * stride_kk, mask=mask_m[:, None], other=0.0).to(DTYPE)
            v = tl.load(v_ptr + offsets_m[:, None] * stride_vm + offsets_d[None, :] * stride_vk, mask=mask_m[:, None], other=0.0).to(DTYPE)
            dy = tl.load(dy_ptr + offsets_m[:, None] * stride_dym + offsets_d[None, :] * stride_dyk, mask=mask_m[:, None], other=0.0).to(DTYPE)

            l2_q = tl.sqrt(tl.sum(q * q, axis=1) + 1e-8)
            l2_k = tl.sqrt(tl.sum(k * k, axis=1) + 1e-8)
            r = l2_q + l2_k + 1.0

            mstate = c_n_vec[None, :] / r[:, None]
            sig_m = tl.sigmoid(mstate)
            g = mstate * sig_m
            o0 = v * (1.0 + g)

            out_var = tl.sum(o0 * o0, axis=1) / head_dim
            inv_rms_out = tl.rsqrt(out_var + 1e-5)
            out_y = o0 * inv_rms_out[:, None]

            mean_dy_y = tl.sum(dy * out_y, axis=1) / head_dim
            do0 = inv_rms_out[:, None] * (dy - out_y * mean_dy_y[:, None])

            dV_from_o0 = do0 * (1.0 + g)
            tl.store(dv_ptr + offsets_m[:, None] * stride_dvm + offsets_d[None, :] * stride_dvk, dV_from_o0.to(dV.dtype.element_ty), mask=mask_m[:, None])

            dg = do0 * v
            dsilu_m = sig_m * (1.0 + mstate * (1.0 - sig_m))
            dmstate = dg * dsilu_m

            dc_n = dmstate / r[:, None]
            dr = - tl.sum(dmstate * c_n_vec[None, :], axis=1) / (r * r)

            # RMSNorm(c) bwd with global c
            mean_dcny = tl.sum(dc_n * c_n_vec[None, :], axis=1) / head_dim
            dc = inv_rms_c_g * (dc_n - c_n_vec[None, :] * mean_dcny[:, None])
            dc = tl.where(mask_m[:, None], dc, 0.0)
            sum_dc_global += tl.sum(dc, axis=0)

            dQ_block = tl.where(l2_q[:, None] > 0.0, dr[:, None] * (q / l2_q[:, None]), 0.0)
            dK_block = tl.where(l2_k[:, None] > 0.0, dr[:, None] * (k / l2_k[:, None]), 0.0)
            tl.store(dq_ptr + offsets_m[:, None] * stride_dqm + offsets_d[None, :] * stride_dqk, dQ_block.to(dQ.dtype.element_ty), mask=mask_m[:, None])
            tl.store(dk_ptr + offsets_m[:, None] * stride_dkm + offsets_d[None, :] * stride_dkk, dK_block.to(dK.dtype.element_ty), mask=mask_m[:, None])

        # Pass 3: propagate dc through V via swiglu with global dv_sw = sum_dc_global
        for start_m in range(0, seq_len, BLOCK_SIZE_M):
            offsets_m = start_m + tl.arange(0, BLOCK_SIZE_M)
            mask_m = offsets_m < seq_len
            v = tl.load(v_ptr + offsets_m[:, None] * stride_vm + offsets_d[None, :] * stride_vk, mask=mask_m[:, None], other=0.0).to(DTYPE)
            sig_v = tl.sigmoid(v)
            v_silu = v * sig_v
            dsilu_v = sig_v * (1.0 + v * (1.0 - sig_v))
            dv_from_vsw = sum_dc_global[None, :] * (dsilu_v * v + v_silu)

            dv_prev = tl.load(dv_ptr + offsets_m[:, None] * stride_dvm + offsets_d[None, :] * stride_dvk, mask=mask_m[:, None], other=0.0)
            dV_block = dv_prev + dv_from_vsw
            tl.store(dv_ptr + offsets_m[:, None] * stride_dvm + offsets_d[None, :] * stride_dvk, dV_block.to(dV.dtype.element_ty), mask=mask_m[:, None])


if _TRITON_AVAILABLE:
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
        def __init__(self):
            super().__init__()

        def forward(self, Q, K, V, causal: bool = True, accum_dtype: torch.dtype | None = None, block_m: int | None = None, num_warps: int | None = None, num_stages: int | None = None):
            return EvoAttentionCausalFunction.apply(Q, K, V, causal, accum_dtype, block_m, num_warps, num_stages)
else:
    class EvoAttentionCausalTriton(nn.Module):
        """CPU/No-Triton fallback that routes to the Torch reference implementation.

        This keeps the same class name so downstream code does not need to branch.
        """

        def __init__(self):
            super().__init__()

        def forward(self, Q, K, V, causal: bool = True, accum_dtype: torch.dtype | None = None, block_m: int | None = None, num_warps: int | None = None, num_stages: int | None = None):  # noqa: ARG002
            ref = EvoAttentionCausalTorch()
            return ref(Q, K, V, causal=causal)


# ------------------ Tests ------------------
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

    with torch.no_grad():
        out_ref = ref(Q, K, V)
        out_trt = trt(Q, K, V)

    diff = (out_ref - out_trt).abs().max().item()
    print(f"Causal forward correctness (max abs diff): {diff:.2e}")
    assert diff < 1e-4, "Causal forward mismatch"


def test_backward_correctness_small():
    if not torch.cuda.is_available():
        print("CUDA not available, skipping causal backward correctness test.")
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

    dq_diff = (Q1.grad - Q2.grad).abs().max().item()
    dk_diff = (K1.grad - K2.grad).abs().max().item()
    dv_diff = (V1.grad - V2.grad).abs().max().item()
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
        out_ref = ref(Q, K, V, causal=False)
        out_trt = trt(Q, K, V, causal=False)

    diff = (out_ref - out_trt).abs().max().item()
    print(f"Non-causal forward correctness (max abs diff): {diff:.2e}")
    assert diff < 1e-4, "Non-causal forward mismatch"


def test_backward_correctness_small_noncausal():
    if not torch.cuda.is_available():
        print("CUDA not available, skipping non-causal backward correctness test.")
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

    dq_diff = (Q1.grad - Q2.grad).abs().max().item()
    dk_diff = (K1.grad - K2.grad).abs().max().item()
    dv_diff = (V1.grad - V2.grad).abs().max().item()
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
                out_ref = ref(Q.float(), K.float(), V.float(), causal=causal).to(dt)
                out_trt = trt(Q, K, V, causal=causal)

            diff = (out_ref - out_trt).abs().max().item()
            print(f"Multi-dtype {str(dt)} causal={causal} max diff: {diff:.2e}")
            tol = 3e-3 if dt in (torch.float16, torch.bfloat16) else 1e-4
            assert diff < tol, "Multi-dtype forward mismatch"


def test_forward_perf():
    if not torch.cuda.is_available():
        print("CUDA not available, skipping causal performance test.")
        return

    device = "cuda"
    dtype = torch.float32
    B, H, L, D = 4, 8, 8192, 64

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
    B, H, L, D = 2, 8, 4096, 64

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
    B, H, L, D = 4, 8, 8192, 64

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

    B, H, L, D = 2, 8, 4096, 64

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

