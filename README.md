# Finding and testing calendar geometry in language models

This project reconstructs the published weekday, month, and year groups in frozen GPT-2-small and ranks all 1,000 groups before reading token labels. Independently trained sparse autoencoders test whether a new dictionary recovers the geometry. In Mistral, we change calendar coordinates at the input month token and measure the effect on the answer.

[Visual results guide](RESULTS.md) · [Jupyter tutorials with bundled data](notebooks/README.md)

The GPT-2 experiment uses the 1,000 groups from the authors' archive at commit `ff8c7eee8f13d78805f590ced6f3fc2227dd1daa` and Joseph Bloom's 24,576-feature SAE at `blocks.7.hook_resid_pre`. The author-provided pickle was converted into [JSON](published-groups.json) after verifying its checksum and list and integer structure. It partitions all 24,576 feature IDs. The reported mean decoder cosine checks are reproduced at 0.6341 for weekdays, 0.5252 for months, and 0.5571 for 20th-century years. [Engels et al.](https://proceedings.iclr.cc/paper_files/paper/2025/hash/d3221cdb27e49d9c1cd35ad254feccfe-Abstract-Conference.html), [Replication repository](https://github.com/JoshEngels/MultiDimensionalFeatures), [Bloom's SAE](https://huggingface.co/jbloom/GPT2-Small-SAEs-Reformatted)

## Blind discovery in GPT-2-small

- A [weekday plane fit on Pile](results/published/calendar-geometry.png) placed 1,363 of 1,367 named weekdays nearest the right weekday angle in the [held-out news text](results/document-controls/calendar-results.csv). These examples came from 466 documents. One document supplied 94.7% of named weekdays in the separate Pile holdout. Month angle accuracy ranged from about 49% to 57% across the three evaluation sources. The [year projection](results/year-geometry/year-geometry.png), fit on PC 3 and PC 4, preserved calendar order in an arc with a mean circular error of 4.4–4.7 years across the three holdouts.
- The [blind ranks](results/published/rank-transfer.png) depended on the sampler, score, and corpus. With a four-point cap per document, the month group ranked fourth on news and first by the [geometric mean of raw ranks across four corpora](results/document-controls/corpus-consensus-ranks.csv). Scoring clusters of projected points put the weekday group first on news and 251st to 672nd elsewhere. Only 2–8 of the top 25 raw candidates transferred to the omitted corpus in [leave-one-corpus-out ranking](results/document-controls/corpus-consensus-transfer.csv).
- [Clustering latent coactivations and decoder directions](results/discovery/fine-groups/clustering-results.json) without labels put all nine published weekday latents and all sixteen month latents into separate, larger groups. The resulting [news evaluation](results/discovery/fine-groups/joint-geometry/calendar-results.csv) gave 17% and 43% angle accuracy in those broader groups. Membership and geometry can be measured separately.
- In a [blind shortlist audit](results/discovery/candidate-audit/candidate-audit.csv), we shuffled sparse coactivations while preserving each point's number of active latents and each latent's nonzero activation values. The raw month score exceeded all 31 controls in five of six samples from news and WikiText test. Candidate 334 also had positive median score differences in both corpora, and its [revealed tokens](results/discovery/candidate-audit/candidate-labels.csv) included `legs` and `feet` in 59 of 128 news documents sampled. The separate [suffix candidate](results/discovery/candidate-271.json) consistently captured `-ing` tokens. Both are candidates for semantic follow-up.
- [Injected rings and filled disks](results/geometry-benchmark/geometry-benchmark.png) test detection at known signal strength in real activation backgrounds. At one-quarter of background RMS, raw persistence achieved 0.80 AUROC and the norm-band score reached 0.84. At half RMS, raw persistence reached 0.998.
- [Controlled calendar formats](results/format-transfer/calendar-format-transfer.csv) used 912 newly written prompts. The fixed weekday plane decoded 100% of full title-case names, 82% in lowercase, 43% as dotted abbreviations, and 32% in uppercase. The fixed month plane decoded 25% of full month names in these prompts. The paired early and late modifiers shifted average angles by less than one degree.

## Independently trained dictionaries

- A [3,072-latent TopK SAE](results/topk/dictionary-comparison.csv) trained on unlabeled GPT-2 activations recovered weekday order in its full reconstruction and across several groups. Its closest single decoder group was active on 40% of named weekdays during discovery. On the Pile holdout, the three closest groups gave 85% conditional accuracy at 98% coverage. The full decoder gave 85% weekday accuracy and 30% month accuracy. The three closest month groups gave 50% at 89% coverage.
- The [block-sparse SAE](results/block/dictionary-comparison.csv) used 3,072 latents in blocks of four and gave 35% weekday and 31% month accuracy with the full decoder on the Pile holdout. Its fit used a lower learning rate and gradient clipping after an unstable training segment.
- The [same-token reconstruction comparison](results/reconstruction-geometry/reconstruction-geometry.csv) projects raw residuals and all full reconstructions into the same reference plane. The 4,500-step TopK SAE had 1.35% unexplained variance across its full validation sample and 43% on named weekday tokens in the Pile holdout, measured relative to their fit mean. The reported document-macro accuracies give each document equal weight.

## Causal tests in Mistral

- The short prompt `Two months after November is` led with a non-month token. The paper's framing made January the top completion. At pre-block layer 12, [the two-dimensional edit](results/steering/framing-summary.json) raised February from 1.93% to 2.06%. A full donor activation patch made February the top token at 79.8%.
- A [larger steering sweep](results/steering/stronger/mistral-steering.json) selected pre-block layer 14 on a separate calibration set. Across 216 evaluation pairs, the two-dimensional edit shifted the target versus original logit gap by 1.29 on average. Six equal-norm random planes averaged 0.012. The paired mean difference had a source-month bootstrap interval of [0.89, 1.73]. The three-month move contributed the largest effects. On the exact one-month edit, February stayed near 2%. The [ellipse and chord](results/steering/stronger/mistral-paths.json) comparison favored different equal-norm paths for its two intermediate targets.
- [Extending the month subspace](results/steering/harmonics/harmonic-interventions.png) with higher Fourier contrasts between month centroids made the target February the top completion at 80.0% on the November example. The 11-dimensional edit recovered a mean logit-gap change of 3.82 for one-month moves, close to 3.86 for the full donor. The same-norm 11-dimensional PCA edit averaged 3.59, and random directions averaged 0.015. A donor edit in the complement of the first two dimensions also moved February to the top. [Intervention data and exact example](results/steering/harmonics/harmonic-metadata.json)
- [Three-layer editing](results/steering/multi-layer-steering/multi-layer-summary.json) improved the one-month logit shift from 0.17 at one layer to 0.33 across three layers with the same combined norm. It reached 2.06% February probability on the exact November example.
- [The published Mistral SAE plane](results/steering/published-sae/sae-plane-steering.json) was fit from 707 active reconstructed points in news text at pre-block layer 8. A one-month edit in that plane improved the target logit gap by 0.147 on average, compared with 0.004 for norm-matched random edits. Editing the reconstruction of the 21 selected features directly improved it by 0.125.
- A [month readout fit on single-step examples](results/steering/calendar-trajectories/calendar-trajectory-summary.csv) reached 12.5% at four moves with cumulative instructions alone and 91.7% when each earlier month was written into the prompt. The [trajectory figure](results/steering/calendar-trajectories/calendar-trajectory.png) uses a separate prompt at each depth. [Goodfire's story-trajectory study](https://www.goodfire.com/research/stories-in-space)

## Run the GPT-2 study

Use Python 3.12 and run from this directory. The scripts choose CUDA, Apple MPS, or CPU in that order. Use a scratch directory with room for activations and model weights. The run described here saved several gigabytes of activation shards, and the optional Mistral checkpoint is about 4 GB.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false
scratch="$PWD/.experiment"
export HF_HOME="$scratch/hf-cache" MPLCONFIGDIR="$scratch/matplotlib"

python extract.py --work-dir "$scratch" download
python extract.py --work-dir "$scratch" extract pile-fit --sequences 2048 --residuals
python rank.py pile-fit --work-dir "$scratch" --max-shards 8 \
  --points 192 --output "$scratch/pilot-rank.csv"
python reveal.py --work-dir "$scratch" --rankings "$scratch/pilot-rank.csv" \
  --output "$scratch/pilot"
python test_study.py
```

The paper computed the original groups on its own activation corpus. This study reuses its groups and weights. The natural-text samples here come from pinned Pile-10k and WikiText-103 releases. Pile documents are assigned to discovery or holdout by a content hash, which keeps exact duplicates in the same split. Each cache uses 128-token sequences and a BOS token. GPT-2 weights are frozen. Only the first seven transformer blocks run, followed by the residual centering used in the original TransformerLens setup. On four CPU comparison sequences, the truncated forward matched the full-model residual at the hook exactly. CPU and MPS had identical SAE firing support on that check. [Pile-10k](https://huggingface.co/datasets/NeelNanda/pile-10k), [WikiText](https://huggingface.co/datasets/Salesforce/wikitext)

```bash
python extract.py --work-dir "$scratch" extract pile-fit --sequences 16384 --residuals
python extract.py --work-dir "$scratch" extract pile-holdout --sequences 6144 --residuals
python extract.py --work-dir "$scratch" extract wiki-train --sequences 6144 --residuals
mkdir -p "$scratch/rankings"

for split in pile-fit pile-holdout wiki-train; do
  stem="$split"
  if [ "$split" = pile-fit ]; then stem=full-fit; fi
  for sampler in balanced uniform; do
    python rank.py "$split" --work-dir "$scratch" --sampling "$sampler" \
      --points 224 --repeats 3 --modes raw unit transverse pca23 shell \
      --output "$scratch/rankings/$stem-$sampler.csv"
  done
done

python reveal.py --work-dir "$scratch" --rankings \
  "$scratch/rankings/full-fit-balanced.csv" \
  "$scratch/rankings/pile-holdout-balanced.csv" \
  "$scratch/rankings/wiki-train-balanced.csv" --output "$scratch/published"
python summarize_ranks.py --rankings "$scratch/rankings" --output "$scratch/published"
```

`rank.py` reads sparse codes and decoder weights. It represents each group's reconstructions in an orthonormal basis of that group's decoder span, preserving pairwise distances. The raw score is the longest one-dimensional persistence interval divided by the 90th percentile of pairwise distance. The other settings test unit normalization, removal of PC 1, use of PC 2 and PC 3, and splitting into three bands of activation magnitude. Balanced sampling allocates examples to live SAE features before filling the remaining slots. With `--controls`, the same transformations are applied to Gaussian draws matched to each sampled cloud's fitted mean and covariance. Topology ranking depends on the sampling procedure and the chosen geometry. [Engels et al., Section 4](https://arxiv.org/html/2405.14860#S4), [Setlur's persistent-homology experiment](https://siddharthsetlur.github.io/perslens_exposition/)

The second refinement tests whether assigning samples by the group's share of the full SAE code surfaces the right geometry. It also tests two alternative scores. `annulus` measures center clearance, radius variation, and angle coverage in a fitted plane. `prototype` computes persistent homology on k-means centroids from PC 2 and PC 3. The fixed settings for the final WikiText test are in the [second refinement protocol](results/second-refinement.txt).

The label reveal uses the exact full names found at active positions. It learns the PCA basis from all active points in the discovery sample and uses weekday or month labels only to orient the plotted plane. Held-out examples use the same orientation. Reported token accuracy is the proportion of labeled points placed nearest their calendar angle, conditional on group activation. A token such as *May* can occur in other senses. The WikiText train file is a separate evaluation corpus in this project.

The Pile holdout's largest document accounts for 94.7% of the labeled weekday tokens and 70.5% of the labeled month tokens. The WikiText weekday examples came from 35 articles. A further run uses the strongest four points per document and group, with the CNN/DailyMail 3.0.0 test articles as an additional source. The [document-control protocol](results/document-control-protocol.txt) records this choice. [CNN/DailyMail dataset](https://huggingface.co/datasets/abisee/cnn_dailymail)

```bash
python extract.py --work-dir "$scratch" download-news
python extract.py --work-dir "$scratch" extract news-test --sequences 6144
python extract.py --work-dir "$scratch" extract wiki-test --sequences 2304
for split in pile-fit pile-holdout wiki-train news-test; do
  python rank.py "$split" --work-dir "$scratch" --sampling balanced --per-document-limit 4 \
    --points 224 --repeats 3 --modes raw pca23 prototype \
    --output "$scratch/rankings/capped-$split.csv"
done
python summarize_document_controls.py --rankings "$scratch"/rankings/capped-*.csv \
  --output "$scratch/corpus-consensus"
python audit_candidates.py --work-dir "$scratch" --discovery-rankings \
    "$scratch/rankings/capped-pile-fit.csv" "$scratch/rankings/capped-pile-holdout.csv" \
    "$scratch/rankings/capped-wiki-train.csv" \
  --evaluation news-test wiki-test --output "$scratch/candidate-audit"
python year_geometry.py --work-dir "$scratch" --rankings \
    "$scratch/rankings/capped-pile-fit.csv" "$scratch/rankings/capped-pile-holdout.csv" \
    "$scratch/rankings/capped-wiki-train.csv" "$scratch/rankings/capped-news-test.csv" \
  --output "$scratch/year-geometry"
python group_from_text.py --work-dir "$scratch" --target-groups 1000 --output "$scratch/groups"
```

The shortlist shuffles support via Curveball trades and permutes the original nonzero weights within each feature. The exceedance counts describe the sampled controls. [Curveball algorithm](https://www.nature.com/articles/ncomms5114)

The optional `pca-adjacent` scorer takes the strongest persistent loop among adjacent pairs of the first five principal components. The `irreducibility` scorer averages a mixture-and-separability score over those planes. It uses 12 bins per axis, a 32-angle grid, and an affine-band search with epsilon 0.1. The comparison used 224 points per group and seven covariance-matched Gaussian controls. This is a small-sample approximation to Appendix B.2. [Engels et al., Appendix B.2](https://arxiv.org/html/2405.14860#A2.SS2)

```bash
for split in pile-fit pile-holdout wiki-train news-test; do
  python rank.py "$split" --work-dir "$scratch" --sampling balanced --per-document-limit 4 \
    --points 224 --modes pca-adjacent irreducibility --controls 7 \
    --output "$scratch/rankings/alternative-$split.csv"
done
python refine_groups.py --work-dir "$scratch" --groups "$scratch/groups/joint-groups.json" \
  --output "$scratch/subgroups"
```

## Train a second dictionary

```bash
model="$scratch/topk"
python train_sae.py --work-dir "$scratch" train --fit-shards 40 \
  --validation-shards 6 --output "$model" --steps 4500
python train_sae.py --work-dir "$scratch" group --model "$model"
for split in pile-fit pile-holdout wiki-train; do
  python train_sae.py --work-dir "$scratch" encode --model "$model" \
    --split "$split" --output-split "topk-$split"
  python rank.py "topk-$split" --work-dir "$scratch" --groups "$model/groups.json" \
    --decoder "$model/decoder.npy" --sampling balanced --points 224 --repeats 3 \
    --modes raw pca23 --output "$scratch/rankings/topk-$split.csv"
done
python compare_dictionary.py --work-dir "$scratch" --model "$model" \
  --published "$scratch/rankings/full-fit-balanced.csv" \
    "$scratch/rankings/pile-holdout-balanced.csv" "$scratch/rankings/wiki-train-balanced.csv" \
  --trained "$scratch/rankings/topk-pile-fit.csv" \
    "$scratch/rankings/topk-pile-holdout.csv" "$scratch/rankings/topk-wiki-train.csv" \
  --output "$scratch/dictionary-comparison"
```

The independent SAE has 3,072 latents and retains up to 32 per example. The block-sparse extension uses `--block-size 4` during training and `--native-blocks` during grouping. It retains up to eight four-coordinate blocks. The comparison chooses learned groups by decoder overlap with the published plane, then reveals the labels to measure calendar order. The block extension was inspired by work on using small groups of coordinates to represent a concept. [Goodfire's SAE manifold work](https://www.goodfire.com/research/can-saes-capture-neural-geometry), [Goodfire's block-sparse featurizers](https://www.goodfire.com/research/bsf-vision)

Use `--loss relative-token` to divide the reconstruction error of each example by its normalized squared residual size, with a floor of 0.05. `--loss relative-sqrt` uses the square root of that quantity. The matched loss comparison uses seeds 7, 11357, and 91452, the first 20 training shards, 2,500 steps, a learning rate of 0.0007, and gradient clipping at 0.5. `compare_reconstructions.py` takes one or more `--model LABEL=PATH` arguments and a list of ranking files with residual-enabled caches.

## Test causal use on Apple Silicon

```bash
python -m pip install -r requirements-apple-steering.txt
python steer_months.py --download --model-dir "$scratch/mistral-4bit" \
  --output "$scratch/steering" --compact-fit --candidate-layers 2 4 6 8 10 12 14 \
  --plane-fit all-rows centroids --calibration-style paper --calibration-offsets 4 9 \
  --test-styles paper paper-after --example-style paper --test-offsets 2 5 7 \
  --donor-shifts 1 -1 3 --random-subspaces 6
python steer_contexts.py --model-dir "$scratch/mistral-4bit" \
  --output "$scratch/framing"
python steer_multiple_layers.py --model-dir "$scratch/mistral-4bit" \
  --calibration "$scratch/steering/mistral-example.json" --output "$scratch/multiple-layers"
python steer_harmonics.py --model-dir "$scratch/mistral-4bit" \
  --selection "$scratch/steering/mistral-example.json" --output "$scratch/harmonics"
python calendar_trajectories.py --model-dir "$scratch/mistral-4bit" \
  --output "$scratch/trajectories"
python steer_published_sae.py --download --work-dir "$scratch" \
  --model-dir "$scratch/mistral-4bit" --sae-dir "$scratch/mistral-sae-layer8" \
  --output "$scratch/sae-plane-steering"
```

The experiment uses the 4-bit MLX conversion of Mistral-7B-v0.1. It fits the two-dimensional month plane using sine and cosine targets and freezes the intervention layer on a separate set of offsets. Patches modify only the month-token residual at the selected layer. Controls include norm-matched random and principal-component directions, the orthogonal complement of the calendar plane, and a full donor activation patch. Confidence intervals resample source months. The interpolation experiment compares a fitted ellipse with its chord at equal fractions and with a norm-matched chord. [Model weights](https://huggingface.co/mlx-community/Mistral-7B-v0.1-q), [Engels et al., Section 5](https://arxiv.org/html/2405.14860#S5), [Goodfire's manifold steering](https://www.goodfire.com/research/manifold-steering)
