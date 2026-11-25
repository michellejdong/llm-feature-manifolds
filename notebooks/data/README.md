# Bundled notebook data

| File | Contents |
| --- | --- |
| [blind-groups.npz](blind-groups.npz) | Up to 128 rows for each of 12 curated GPT-2 groups. Anonymous keys end in `_codes` for the selected latents and `_decoder` for their directions. The numerical data come from the Pile discovery split. |
| [group-reveal.csv](group-reveal.csv) | Published group IDs and brief descriptions. Open it after ranking in the first notebook. |
| [weekday-reveal.npz](weekday-reveal.npz) | Unlabeled reconstruction coordinates from Pile in `fit_cloud`, and 1,367 CNN/DailyMail test examples in `news_points`. `weekday_id` indexes `weekday_names`. A common orthonormal basis preserves distances within the group decoder span. |
| [steering-examples.csv](steering-examples.csv) | 864 recorded Mistral prompt and edit cases. The key columns are `style`, `offset` in the arithmetic prompt, `source_month`, `donor_shift`, `condition`, and `dimensions`. The target probability is normalized over the full vocabulary. |

The small collection was selected to teach the protocol. The full [ranking tables](../../results/rankings/) score all published groups. [Export script](../export_examples.py) · [Original feature groups](https://github.com/JoshEngels/MultiDimensionalFeatures) · [CNN/DailyMail](https://huggingface.co/datasets/abisee/cnn_dailymail)
