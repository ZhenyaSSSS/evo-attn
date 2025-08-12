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

Функциональный API для уже спроецированных тензоров `q, k, v` формы `(B, H, L, D)`:

```python
import torch
from evo_attn import evo_attention

B, H, L, D = 2, 8, 4096, 64
q = torch.randn(B, H, L, D, device='cuda')
k = torch.randn(B, H, L, D, device='cuda')
v = torch.randn(B, H, L, D, device='cuda')
y = evo_attention(q, k, v, causal=True)
```

Модуль с линейными проекциями Q/K/V (drop-in блок):

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
- Ядра Triton с численно стабильными редукциями и компенсированными префикс‑суммами
- Автоматический PyTorch‑фоллбек при отсутствии Triton/CUDA
- Опциональная `attention_mask` для паддинга
- Поддержка fp16/bf16/fp32; контрольной точности аккумулирования

### Совместимость

- Python ≥ 3.10, PyTorch ≥ 2.1
- Triton ≥ 2.1 (рекомендуется Linux). На Windows используйте WSL2 или полагайтесь на фоллбек.

### Лицензия

Apache-2.0 (см. файл `LICENSE`).

