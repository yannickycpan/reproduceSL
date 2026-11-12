# Scoped experiment identity and reproducibility

**One branch:** `fullTD`. Scientific source identities are no longer
monolithic for the two application datasets. Each application candidate
records (i) its exact immutable scientific configuration, (ii) the prepared
dataset identity, and (iii) a source-code hash constructed from **only
its dependencies**.

These dependencies include shared train/metrics/source-neighbor utilities,
the selected dataset's sampling and mean-network construction, relevant
sections of the unchanged model/data implementations, the chosen objective,
and the selected baseline or TD head. Thus modifying the METR-LA sampler
does not change Electricity code hashes; changing the pairwise TD class
does not change MSE code hashes. Changes in genuinely common functions,
such as metric aggregation or the logical-batch backward implementation,
correctly invalidate all users of those functions.

FD001 retains its original byte-for-byte source hash and is outside this
application identity refactor. The application identity layout itself is
new, so old Electricity/application training checkpoints from the
monolithic-hash implementation are not directly resume-compatible.
Keep old results backed up; do not overwrite or relabel them.

## Source fractions

The training dataset and its eligible TD partner universe remain complete.
For a fraction below 100%, sources are drawn iid with replacement each
round. The 100% special case uses the **original deterministic full
random reshuffling** at every epoch. Every fraction runs 100 rounds with
the original Graph WaveNet, masks, 207 sensors, optimizer, loss definitions,
learning-rate grids, five seeds and logical batch 256.

For N around 23,967, 10% leads to 2,397 draws per round and
1,000 optimizer updates over 100 rounds; 25% leads to 5,992 draws,
2,400 updates; 100% restores 23,967 draws and 9,400 updates.
METR-LA validation frequency is 100/250/500 updates for
10%/25%/100% respectively. The last quarter of positive-update validation
measurements still determines the selected HP; tests remain held out.

The 10% setup is a **compute-budget comparison**, not 10%-label learning.
The real wall-clock speedup must be measured on the server; the drop in
updates is not an exact latency guarantee. The low budget is scientifically
useful only if convergence/optimization adequacy is checked.

## Simple sibling folders

```text
~/scratch/fulltd/
  fd001_results/
  fd001_outputs/
  electricity_results/
  electricity_outputs/
  metr_la_f10pct_results/
  metr_la_f10pct_outputs/
```

Set `FULLTD_METR_LA_SOURCE_FRACTION=0.25` or `1.0` to use
`metr_la_f25pct_results` or `metr_la_f100pct_results` and matching
outputs. Do not set the old `FULLTD_METR_LA_ROOT` root when using the new
default; `result_paths` actively rejects the old untagged root.

## Migration and launch (AMD CPU)

**First stop the original METR-LA dispatcher and workers gracefully, and
back up the old results**. Do not pull new code into an actively running
application job. The tools do not delete old results automatically.

```bash
cd ~/reproduceSL
git switch fullTD
git pull --ff-only origin fullTD
# Default fraction is 10%, no extra launcher/branch required
$HOME/scratch/fulltd/.venv-fulltd/bin/python -m pytest -q tests dscpuscripts
bash dscpuscripts/run_metr_la.sh --dry-run
nohup bash dscpuscripts/run_metr_la.sh --run --workers 128 --adaptive \
  > "$HOME/metr_la_f10pct.log" 2>&1 < /dev/null &
```

After all 525 candidates finish or have documented numerical failures:

```bash
bash dscpuscripts/analyze_standard.sh metr_la check
bash dscpuscripts/analyze_standard.sh metr_la run
```

The `--status` mode uses its own default displayed worker count; that
number is not the running dispatcher's actual `--workers` cap.
