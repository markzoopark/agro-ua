# EDA Полтавська область

## Тренди 2010–2024
![Тренди урожайності та факторів](reports/figures/poltava_trends.png)

## Кореляційний аналіз
### Pearson (Yield, Yield_anom)
| Factor                |   Pearson_Yield |   Pearson_Yield_anom |
|-----------------------|-----------------|----------------------|
| Area_ha               |           0.647 |               -0.109 |
| N_kg_ha               |           0.662 |               -0.027 |
| P2O5_kg_ha            |          -0.309 |               -0.209 |
| K_kg_ha               |          -0.306 |               -0.149 |
| Mineral_treated_share |           0.295 |                0.009 |
| Org_kg_ha_or_share    |          -0.218 |                0.134 |
| Irrig_m3_ha           |          -0.086 |               -0.359 |
| Irrig_mm              |          -0.086 |               -0.359 |

### Spearman (Yield, Yield_anom)
| Factor                |   Spearman_Yield |   Spearman_Yield_anom |
|-----------------------|------------------|-----------------------|
| Area_ha               |            0.49  |                -0.146 |
| N_kg_ha               |            0.728 |                 0.058 |
| P2O5_kg_ha            |           -0.248 |                -0.239 |
| K_kg_ha               |           -0.24  |                -0.188 |
| Mineral_treated_share |            0.354 |                -0.055 |
| Org_kg_ha_or_share    |           -0.236 |                 0.168 |
| Irrig_m3_ha           |           -0.081 |                -0.377 |
| Irrig_mm              |           -0.081 |                -0.377 |

![Кореляційна теплокарта](reports/figures/correlation_heatmap.png)

## Автокореляція урожайності
![Автокореляція урожайності](reports/figures/yield_autocorrelation.png)

*Автоматичний звіт згенеровано на основі `data/processed/agrostats_poltava_features.parquet`.*
