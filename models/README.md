# Models and Data

## Dataset

All evaluation experiments use the AudioSet evaluation split. 50 clips are sampled from each of 22 classes (1,100 clips total). Full list can be found at the bottom of this README.md

AudioSet is not redistributed here. The evaluation split can be downloaded from the [official AudioSet page](https://research.google.com/audioset/download.html).

## Models

### AST

- Checkpoint: `MIT/ast-finetuned-audioset-10-10-0.4593`
- Source: [HuggingFace](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593)
- Loaded automatically via the HuggingFace `transformers` library — no manual download required.

### PANNs CNN14 (no SpecAugment) and PANNs CNN14 (+SpecAugment)

Both variants use the CNN14 architecture from [PANNs](https://github.com/qiuqiangkong/audioset_tagging_cnn), trained on AudioSet. The sole training difference is whether SpecAugment was applied.

Checkpoints will be deposited to HuggingFace soon. In the meantime, contact the authors or train your own variants using the PANNs codebase with and without the SpecAugment flag.

The `audioset_tagging_cnn` repository must be cloned and its `pytorch/` directory added to the Python path before loading either PANNs wrapper. See the comments in `src/partial_masks/model_wrappers/panns_no_specaug.py`.