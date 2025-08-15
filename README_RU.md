## EvoAttention (Triton + Torch fallback)

Высокопроизводительный EvoAttention с причинным (causal) и не причинным (non-causal) режимами на Triton с автоматическим запасным путём на PyTorch. Численно совпадает с референсной реализацией и обеспечивает ускорение вперёд на 3–5× на длинных последовательностях.

Русская версия этого README. English version is in `README.md`.

### Установка

```bash
pip install -U pip
pip install .            # установка из корня репозитория
# Triton (опционально; лучше на Linux с CUDA):
pip install 'evo-attn[triton]'
```

Примечание для Windows: Triton в первую очередь поддерживается на Linux. Для лучшего результата используйте WSL2. Если Triton недоступен, автоматически используется PyTorch-фоллбек.

### Быстрый старт

Функциональный API принимает только `v` формы `(B, H, L, D)`; все проекции и гейты внутри:

```python
import torch
from evo_attn import evo_attention

B, H, L, D = 2, 8, 4096, 64
v = torch.randn(B, H, L, D, device='cuda')
y = evo_attention(v, causal=True)
```

Модульный API `(B, L, E) → (B, L, E)`; все линейные операции внутри:

```python
import torch
from evo_attn import EvoAttention

B, H, L, D = 2, 8, 4096, 64
E = H * D
x = torch.randn(B, L, E, device='cuda')
attn = EvoAttention(embed_dim=E, num_heads=H).cuda()
y = attn(x, causal=True)
```

Маска внимания `attention_mask` поддерживается форм в виде `(B, L)`, `(B, 1, L)` или `(B, 1, L, 1)`; 1.0 — валидный токен.

### Особенности

- Causal и non-causal режимы
- Префикс‑суммы в Triton; остальное под капотом на быстрых cuBLAS/cuDNN
- Передаёте только Value; Q/K проекции и гейты обрабатываются внутри
- Автоматический PyTorch‑фоллбек при отсутствии Triton/CUDA
- Опциональная `attention_mask` для паддинга
- Поддержка fp16/bf16/fp32

### Параметры выполнения (Runtime)

- `EVO_BACKEND=auto|triton|torch` — выбор бэкенда (по умолчанию `auto`). При отсутствии Triton автоматически используется PyTorch.
- `EVO_AUTOTUNE=1` — однократная быстрая автотюнинг‑настройка параметров запуска префикс‑ядра (кэшируется).
- Аргумент `accum_dtype` в API — можно передать `torch.float64` на причинном пути для максимальной численной стабильности (если железо поддерживает). При fp16/bf16 редукции по умолчанию выполняются в fp32.

### Совместимость

- Python ≥ 3.10, PyTorch ≥ 2.1
- Triton ≥ 2.1 (рекомендуется Linux). На Windows используйте WSL2 или полагайтесь на фоллбек.

### Лицензия

Apache-2.0 (см. файл `LICENSE`).

