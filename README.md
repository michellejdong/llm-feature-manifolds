# Finding and testing calendar geometry in language models

This project reconstructs the published weekday and month feature groups in frozen GPT-2-small, ranks all published groups by their activation geometry, and tests whether their calendar order transfers across documents. The causal experiment fits a calendar plane in Mistral and measures what happens when the model receives a perturbation confined to that plane.

The GPT-2 experiment uses the 1,000 groups from the authors' archive at commit `ff8c7eee8f13d78805f590ced6f3fc2227dd1daa` and Joseph Bloom's 24,576-feature SAE at `blocks.7.hook_resid_pre`. The author-provided pickle was converted into [JSON](published-groups.json) after verifying its checksum and list and integer structure. It partitions all 24,576 feature IDs. The reported mean decoder cosine checks are reproduced at 0.6341 for weekdays, 0.5252 for months, and 0.5571 for 20th-century years. [Engels et al.](https://proceedings.iclr.cc/paper_files/paper/2025/hash/d3221cdb27e49d9c1cd35ad254feccfe-Abstract-Conference.html), [Replication repository](https://github.com/JoshEngels/MultiDimensionalFeatures), [Bloom's SAE](https://huggingface.co/jbloom/GPT2-Small-SAEs-Reformatted)

## Results and files

- [The larger calendar figure](results/published/calendar-geometry.png) shows a projection fit on 2,050,056 Pile tokens and transferred to 780,400 held-out Pile tokens and 779,920 WikiText tokens. All 679 active, explicitly named weekdays in the Pile holdout and all 71 in WikiText landed nearest their weekday's angle. The same test put 56.7% and 49.4% of active month tokens nearest their month angle. The month centroid errors averaged 14.3° and 13.5° on the two evaluation samples.
- [Blind-rank transfer](results/published/rank-transfer.png) shows that the scorer's ranking depended on the text source and sampling method. The selected raw balanced scorer placed weekdays at 533, 352, and 460 on discovery, Pile holdout, and WikiText. It placed months at 31, 119, and 38. Uniform raw sampling placed months sixth on WikiText and 496th on the Pile holdout.
- [The initial calendar figure](results/pilot/calendar-geometry.png) uses 254,801 Pile tokens. Raw persistent homology ranked weekdays 53rd and months 60th. Unit normalization moved them to 326th and 584th.
- [Independent SAE comparison](results/topk/dictionary-comparison.csv) shows how the weekday coordinates were distributed in the 3,072-latent TopK model. Its closest single decoder group activated on 40.4% of weekday-name tokens during discovery and covered an incomplete set of weekdays. Combining the three closest groups yielded 85.0% conditional weekday angle accuracy on the Pile holdout at 98.4% coverage. The complete learned dictionary yielded 84.8% on that same holdout. The three closest groups for months reached 50.2% accuracy at 89.2% coverage.
- [Mistral prompt framing](results/steering/framing-summary.json) compares 24 prompts per frame. The paper's wording gave 24 correct top tokens, compared with zero for the short unframed wording. On the exact November example in the paper's frame, the plane-only edit raised February's probability from 1.93% to 2.06%. A full donor activation patch raised it to 79.8% and made February the top token. The plane-only edit shifted the target logit gap by 0.43 on average in the paper's frame, versus 0.008 for six norm-matched random planes.
- [The ranking protocol](results/refinement-protocol.txt) records the larger held-out experiment. The [rankings folder](results/rankings/) contains the complete tables and run metadata.
- `extract.py` pins and downloads assets and caches activations. `rank.py` assigns scores. `reveal.py` opens the token labels, records ranking checksums, and measures calendar order in a fixed coordinate system.
- `train_sae.py` trains an independent TopK or block-sparse SAE. `compare_dictionary.py` measures how much of the published plane its decoder groups recover. `steer_months.py` runs the Mistral intervention.

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

## Test causal use on Apple Silicon

```bash
python -m pip install -r requirements-apple-steering.txt
python steer_months.py --download --model-dir "$scratch/mistral-4bit" \
  --output "$scratch/steering"
python steer_contexts.py --model-dir "$scratch/mistral-4bit" \
  --output "$scratch/framing"
```

The experiment uses the 4-bit MLX conversion of Mistral-7B-v0.1. It fits the two-dimensional month plane using sine and cosine targets, calibrates the intervention layer on a separate prompt family and offsets, then freezes the plane and layer for the test. Patches modify only the month-token residual at the selected layer. Controls include equally large perturbations in eight random two-dimensional subspaces, a top principal-component direction, and a full donor activation patch. Confidence intervals resample source months. The interpolation experiment compares a fitted elliptical path with its chord at equal fractions and with a norm-matched chord. [Model weights](https://huggingface.co/mlx-community/Mistral-7B-v0.1-q), [Engels et al., Section 5](https://arxiv.org/html/2405.14860#S5), [Goodfire's manifold steering](https://www.goodfire.com/research/manifold-steering)
