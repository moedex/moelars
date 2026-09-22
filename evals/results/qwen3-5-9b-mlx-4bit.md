# mlx-community/Qwen3.5-9B-MLX-4bit

Backend `mlx`, 200 test rows per config, calibration fitted on 200 validation rows. MoeLAR 0.0.1. Jev column quoted from jev-bench's published jev-1.13.0 run on full splits. Model load 1.5s. Peak memory 7.1 GB.

| config | prim | K | acc raw | acc cal | ECE raw | ECE cal | Brier raw | Brier cal | cov@5% | ms/row | Jev acc | Jev ECE | Jev Brier |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| banking77 | choice | 77 | 0.695 | 0.695 | 0.093 | 0.106 | 0.463 | 0.469 | 0.17 | 743 | 0.796 | 0.095 | 0.317 |
| boolq | noul | 2 | 0.870 | 0.870 | 0.131 | 0.053 | 0.228 | 0.196 | 0.62 | 1052 | 0.917 | 0.021 | 0.061 |
| sst5 | score | 5 | 0.475 | 0.475 | 0.148 | 0.062 | 0.684 | 0.665 | 0.03 | 186 | 0.565 | 0.19 | 0.618 |
| clinc150 | choice | 151 | 0.780 | 0.780 | 0.062 | 0.122 | 0.350 | 0.362 | 0.42 | 716 | 0.893 | 0.033 | 0.159 |
| massive | choice | 60 | 0.705 | 0.705 | 0.099 | 0.086 | 0.412 | 0.411 | 0.41 | 392 | 0.808 | 0.09 | 0.295 |
| ledgar | choice | 100 | 0.665 | 0.665 | 0.072 | 0.088 | 0.478 | 0.478 | 0.35 | 420 | 0.751 | 0.117 | 0.374 |
| go_emotions | choice | 28 | 0.230 | 0.230 | 0.277 | 0.051 | 0.536 | 0.416 | 0.01 | 362 | 0.282 | 0.384 | 1.04 |
| mmlu | choice | 4 | 0.710 | 0.710 | 0.102 | 0.069 | 0.393 | 0.391 | 0.20 | 184 | 0.923 | 0.027 | 0.124 |
| arc_challenge | choice | 4 | 0.910 | 0.910 | 0.147 | 0.070 | 0.142 | 0.115 | 0.93 | 157 | 0.979 | 0.01 | 0.037 |
| mnli | choice | 3 | 0.805 | 0.805 | 0.127 | 0.085 | 0.311 | 0.290 | 0.43 | 172 | 0.883 | 0.032 | 0.176 |
| chaosnli | choice | 3 | 0.670 | 0.670 | 0.069 | 0.079 | 0.109 | 0.139 | 0.12 | 169 | 0.615 | 0.222 | 0.583 |

## Macro averages over the configs above (calibrated)

| scope | n | acc | ECE | Brier | Jev acc | Jev ECE | Jev Brier |
|---|---|---|---|---|---|---|---|
| all | 11 | 0.683 | 0.079 | 0.357 | 0.765 | 0.111 | 0.344 |
| choice | 9 | 0.686 | 0.084 | 0.341 | 0.77 | 0.112 | 0.345 |
| score | 1 | 0.475 | 0.062 | 0.665 | 0.565 | 0.19 | 0.618 |
| noul | 1 | 0.87 | 0.053 | 0.196 | 0.917 | 0.021 | 0.061 |
