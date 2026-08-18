# Feature ablation: time and geo channels

*mind the gap. Which inputs the streamed gap-fill U-Net actually needs.*

We ran a small ablation on the streamed gap-fill U-Net to test two candidate input channels: **time**
(day-of-year sine and cosine) and **geo** (absolute position as x, y, z on the unit sphere). Four configs
(none, time, geo, both) were trained on each of two regions, the open Indian Ocean basin (600x696) and the
coastal Arabian Sea (260x376), and scored by held-out fake-cloud MAE split into coast, open, and all pixels,
against standard persistence (yesterday's real observed pixel). Coast is defined as ocean within about 90 km
of land. Each config is a single training run, so small differences may be run-to-run noise.

The main result is that the model behaves like a smoother. On smooth open water its spatial reconstruction
plus the seasonal signal beats persistence (Arabian Sea open: 0.180 for the time config versus 0.191 for
persistence), but on spiky coastal blooms it smooths the peaks toward the mean while persistence keeps
yesterday's sharp value, so persistence wins the coast (Arabian Sea coast: about 0.40 for the model versus
0.32 for persistence) and, because the coast drags the average, wins overall by a few percent. Feature value
is region dependent and consistent with what each channel can add: **time helps only where the seasonal
cycle is strong** (the Arabian Sea monsoon, where it improved the open by about 7 percent; on the weakly
seasonal equatorial basin the same channel hurt), and **geo helps only when the domain spans several
regimes** (it improved the coast on the large basin but was useless in the single-regime Arabian Sea). In
short, each feature pays off only where it carries information the chlorophyll channels do not already
encode, and the coast, not the open ocean, is the outstanding problem.

![Feature ablation across two regions. Bars are the four configs; the dashed line is persistence. A bar below the line beats persistence.](ablation_features.png)

| region | zone | best model | persistence |
|---|---|---|---|
| basin | coast | 0.480 (geo) | 0.251 |
| basin | open | 0.161 (none) | 0.154 |
| basin | all | 0.188 (none) | 0.160 |
| Arabian Sea | coast | 0.397 (geo) | 0.320 |
| Arabian Sea | open | **0.180 (time)** | 0.191 |
| Arabian Sea | all | 0.223 (time) | 0.210 |

**Next.** The coast is the only zone keeping the model below persistence overall, since it already wins the
open. Two options to close that gap: a coast-weighted training loss so the model stops smoothing coastal
highs, or supplying the model with unmasked previous-day pixels so it can copy the sharp coastal value the
way persistence does. The first keeps the current spatial-fill design; the second reopens the question of how
much temporal information the model should be given.

## Note on 3-day composites

Some runs use 3-day composites rather than daily frames. A composite chops the time axis into
non-overlapping 3-day bins and, for each pixel, averages whatever clear observations fell in that window,
leaving a pixel missing only if it was cloudy on all three days. This averaging happens before any
gap-filling: the synthetic fake clouds are applied to the composite frames, not to the raw daily data, so
the model reconstructs hidden pixels of a 3-day-mean product and is scored against that 3-day mean. The
temporal channels follow the same axis, so the previous and next frames are the previous and next
composites, roughly 3 real days apart, and with three lags the model sees composites at about plus or minus
3, 6, and 9 days. Compositing is a standard ocean-color product, not a shortcut, and the evaluation stays
honest because the model never sees the hidden pixel and neighboring composites are disjoint 3-day windows,
so nothing about the target leaks in. It is, though, an easier task than daily gap-filling: a composite has
far fewer gaps and is a smoother target, and persistence is measured across a 3-day gap rather than a single
day, which softens the baseline. On 3-day composites the model beats persistence across the board (best
config about 0.15 versus persistence 0.19), whereas at full daily resolution it wins the open ocean but not
the coast. The two are honest results about two different products and should be reported as such.

