import jax
from jax import lax
import jax.numpy as jnp
import numpy as np
import einops

import flax
from flax import linen as nn

from typing import Sequence, Optional
from functools import partial


def _init_conv2d(key, shape, dtype=jnp.float32):
    fan_in = np.prod(shape[:-1])
    k = np.sqrt(1 / fan_in)
    return jax.random.uniform(key, shape, dtype, minval=-k, maxval=k)


class MeanShift(nn.Module):
    rgb_range: int
    rgb_mean: Sequence[int] = (0.4488, 0.4371, 0.4040)
    rgb_std: Sequence[int] = (1.0, 1.0, 1.0)
    sign: int = -1

    @nn.compact
    def __call__(self, x):
        std = jnp.array(self.rgb_std).reshape(1, 1, 1, 3)
        kernel = self.variable(
            'mean_shift', 'kernel', lambda k: self.sign * jnp.eye(3).reshape(1, 1, 3, 3) / std
        )
        bias = self.variable(
            'mean_shift', 'bias',
            lambda k: self.sign * jnp.array(self.rgb_mean).reshape(1, 1, 1, 3) * self.rgb_range / std
        )
        return lax.conv_general_dilated(
            x, lax.stop_gradient(kernel), (1, 1), "VALID", dimension_numbers=('NHWC', 'HWIO', 'NHWC')
        ) + lax.stop_gradient(bias)


class ShiftConv2D0(nn.Module):
    n_filters: int

    @staticmethod
    def _kernel_mask_init(shape: Sequence[int], dtype=jnp.float32):
        mask = np.zeros(shape)
        g = shape[-2] // 5
        mask[1, 2, :1*g, :] = 1.
        mask[1, 0, 1*g:2*g, :] = 1.
        mask[2, 1, 2*g:3*g, :] = 1.
        mask[0, 1, 3*g:4*g, :] = 1.
        mask[1, 1, 4*g:, :] = 1.
        return jnp.array(mask, dtype=dtype)

    @nn.compact
    def __call__(self, inputs: jnp.ndarray, **kwargs) -> jnp.ndarray:
        kernel = self.param('kernel', lambda rng, shape: _init_conv2d(rng, shape, dtype=jnp.float32), (3, 3, inputs.shape[-1], self.n_filters))
        bias = self.param('bias', lambda rng, shape: jnp.zeros(shape, dtype=jnp.float32), (1, 1, 1, self.n_filters,))
        mask = self.param('mask', lambda rng, shape: self._kernel_mask_init(shape=shape, dtype=jnp.float32), kernel.shape)
        return lax.conv_general_dilated(
            inputs, kernel * lax.stop_gradient(mask), (1, 1), "SAME", dimension_numbers=('NHWC', 'HWIO', 'NHWC')
        ) + bias


class ShiftConv2D1(nn.Module):
    n_filters: int

    @staticmethod
    def _mask_init(shape: Sequence[int]):
        mask = np.zeros(shape, dtype=np.float32)
        g = shape[-1] // 5
        mask[1, 2, :, :1*g] = 1.
        mask[1, 0, :, 1*g:2*g] = 1.
        mask[2, 1, :, 2*g:3*g] = 1.
        mask[0, 1, :, 3*g:4*g] = 1.
        mask[1, 1, :, 4*g:] = 1.
        return jnp.array(mask)

    @nn.compact
    def __call__(self, inputs: jnp.ndarray, **kwargs) -> jnp.ndarray:
        n_filters_in = inputs.shape[-1]
        kernel = self.param(
            'kernel', lambda rng, shape: self._mask_init(shape), (3, 3, 1, self.n_filters)
        )
        inputs = lax.conv_general_dilated(
            inputs, lax.stop_gradient(kernel), (1, 1), "SAME",
            dimension_numbers=('NHWC', 'HWIO', 'NHWC'), feature_group_count=n_filters_in
        )
        inputs = nn.Conv(
            self.n_filters, (1, 1), padding='SAME', kernel_init=_init_conv2d
        )(inputs)
        return inputs


class LFE(nn.Module):  # Local Feature Extraction
    exp_ratio: int
    shift_conv: nn.Module = ShiftConv2D1
    act: callable = nn.relu

    @nn.compact
    def __call__(self, inputs: jnp.ndarray, **kwargs) -> jnp.ndarray:
        n_filters_in = inputs.shape[-1]
        inputs = self.shift_conv(n_filters_in * self.exp_ratio)(inputs)
        inputs = self.act(inputs)
        inputs = self.shift_conv(n_filters_in)(inputs)
        return inputs


class GMSA(nn.Module):  # Group-wise Multi-scale Self-Attention
    shifts: int
    window_sizes: Sequence[int]
    calc_attn: bool

    def calc_attn_path(self, x: jnp.ndarray, window_size:int) -> jnp.ndarray:
        b, h, w, c = x.shape
        if self.shifts > 0:  # Why shift by window_size? Not self.shifts?
            x = jnp.roll(x, shift=(-window_size // 2, -window_size // 2), axis=(1, 2))
        qv = einops.rearrange(
            x, 'b (h dh) (w dw) (qv c) -> qv (b h w) (dh dw c)', dh=window_size, dw=window_size, qv=2
        )
        q, v = qv
        atn = jax.nn.softmax(jnp.matmul(q, q.T), axis=-1)
        y = jnp.matmul(atn, v)
        y = einops.rearrange(
            y, '(b h w) (dh dw c) -> b (h dh) (w dw) (c)',
            dh=window_size, dw=window_size, h=h // window_size, w=w // window_size
        )
        if self.shifts > 0:
            y = jnp.roll(y, shift=(window_size // 2, window_size // 2), axis=(1, 2))
        return y

    def prev_attn_path(self, x: jnp.ndarray, window_size: int, prev_attn: jnp.ndarray) -> jnp.ndarray:
        b, h, w, c = x.shape
        if self.shifts > 0:
            x = jnp.roll(x, shift=(-window_size // 2, -window_size // 2), axis=(1, 2))
        v = einops.rearrange(
            x, 'b (h dh) (w dw) c -> (b h w) (dh dw) c', dh=window_size, dw=window_size
        )
        y = jnp.matmul(prev_attn, v)
        y = einops.rearrange(
            y, '(b h w) (dh dw c) -> b (h dh) (w dw) (c)',
            dh=window_size, dw=window_size, h=h // window_size, w=w // window_size
        )
        if self.shifts > 0:
            y = jnp.roll(y, shift=(window_size // 2, window_size // 2), axis=(1, 2))
        return y

    @nn.compact
    def __call__(
            self, inputs: jnp.ndarray, training: bool, prev_attns: Optional[Sequence[jnp.ndarray]] = None, **kwargs
    ) -> Sequence[jnp.ndarray]:
        n_filters_in = inputs.shape[-1]
        inputs = nn.Conv(
            n_filters_in * 2 if self.calc_attn else n_filters_in, (1, 1), padding='SAME', kernel_init=_init_conv2d
        )(inputs)
        inputs = nn.BatchNorm()(inputs, use_running_average=not training)  # BatchNorm?
        inputs_s = jnp.split(inputs, 3, axis=-1)

        ys = []

        if self.calc_attn:
            attns = []
            for w, x in zip(self.window_sizes, inputs_s):  # I wanted to use vmap here, but It didn't work.
                y, atn = self.calc_attn_path(x, w)
                ys.append(y)
                attns.append(atn)
            prev_attns = attns

        else:
            for w, x, atn in zip(self.window_sizes, inputs_s, prev_attns):
                y = self.prev_attn_path(x, w, atn)
                ys.append(y)

        y = jnp.concatenate(ys, axis=-1)
        y = nn.Conv(
            n_filters_in, (1, 1), padding='SAME', kernel_init=_init_conv2d
        )(y)

        return y, prev_attns


class ELAB(nn.Module):  # Efficient Long-range Attention Block
    shared_depth: int
    exp_ratio: int
    shifts: int
    window_sizes: Sequence[int]
    calc_attn: bool

    def setup(self) -> None:
        self.ltes = [LFE(self.exp_ratio, self.shift_conv) for _ in range(self.shared_depth + 1)]
        self.gmsas = [
            GMSA(self.shifts, self.window_sizes, i == 0) for i in range(self.shared_depth + 1)
        ]

    def __call__(self, inputs: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        prev_attns = None
        for lte, gmsa in zip(self.ltes, self.gmsas):
            inputs = inputs + lte(inputs)
            residual, prev_attns = gmsa(inputs, training=training, prev_attns=prev_attns)
            inputs = inputs + residual
        return inputs
