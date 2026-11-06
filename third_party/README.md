# Third-party notebooks

The model builders apply text patches to public notebooks written by other people. Those notebooks
are not copied here. Fetch them from Kaggle into these exact paths:

```bash
kaggle kernels pull pilkwang/rsna-knee-baseline-v1 -p third_party/pilkwang_baseline_v1 -m
mv third_party/pilkwang_baseline_v1/kernel-metadata.json \
   third_party/pilkwang_baseline_v1/original-kernel-metadata.json

kaggle kernels pull pjmathematician/rsna-knee-d4-blend -p third_party/pjmathematician_rsna-knee-d4-blend -m
```

The patches assert on exact text, so a newer version of a notebook can make a build stop with a
clear error. These are the SHA-256 hashes of the versions this project used:

| Notebook | SHA-256 |
|---|---|
| `rsna-knee-baseline-v1.ipynb` | `b32a9155fcc73c519e75e78c08e07d30393efc591c2798a3e40ca92f39832bb8` |
| `rsna-knee-d4-blend.ipynb` | `6fd1a3d61a15ad5922972fb6589d372a88d428442950d21740f304ae77fa6b72` |

Credit: [pilkwang/rsna-knee-baseline-v1](https://www.kaggle.com/code/pilkwang/rsna-knee-baseline-v1)
by Pilkwang Kim, and [pjmathematician/rsna-knee-d4-blend](https://www.kaggle.com/code/pjmathematician/rsna-knee-d4-blend)
together with the public model authors it builds on.
