"""PANNs CNN14 with SpecAugment — custom-trained checkpoint."""

import platform
import os
import sys

import torch as tt
import pandas as pd

# audioset_tagging_cnn must be cloned and its pytorch/ directory added to the path.
# See: https://github.com/qiuqiangkong/audioset_tagging_cnn
sys.path.insert(0, os.path.join(os.path.expanduser("~"), "audioset_tagging_cnn", "pytorch"))
from models import Cnn14

classes_num = 527

if platform.uname().system == "Darwin":
    device = tt.device("cpu")
else:
    device = tt.device("cuda")

CHECKPOINT = os.path.join(
    os.path.expanduser("~"), "checkpoints", "panns", "cnn14_specaug", "600000_iterations.pth"
)


class PannsModel:
    model_id = "PANNs/CNN14_specaug_trained"

    def __init__(self):
        self.model = Cnn14(sample_rate=32000, window_size=1024, hop_size=320,
                           mel_bins=64, fmin=50, fmax=14000, classes_num=classes_num)
        checkpoint = tt.load(CHECKPOINT, map_location='cpu', weights_only=False)
        self.model.load_state_dict(checkpoint['model'])
        self.model.to(device)
        self.model.eval()

        csv_path = os.path.join(os.path.expanduser("~"), "checkpoints", "panns", "panns_og_kong", "class_labels_indices.csv")
        df = pd.read_csv(csv_path)
        self.labels = dict(zip(df["index"], df["display_name"]))
        print(f"Loaded: {self.model_id} from {CHECKPOINT}")

    def _forward_from_mel(self, log_mel_batch):
        """
        Inject log mel via a forward hook on logmel_extractor; one forward pass.
        """
        model = self.model
        if hasattr(model, 'module'):
            model = model.module
        L = log_mel_batch.shape[0]

        injection = tt.tensor(log_mel_batch, dtype=tt.float32).to(device)
        injection = injection.unsqueeze(1).permute(0, 1, 3, 2)  # (L, 1, T, 64)

        handle = model.logmel_extractor.register_forward_hook(
            lambda module, inp, out: injection
        )
        try:
            dummy = tt.zeros(L, 32000, dtype=tt.float32).to(device)
            with tt.no_grad():
                output_dict = model(dummy, None)
        finally:
            handle.remove()
        return output_dict

    def _to_predictions(self, output):
        results = []
        for i in range(output.shape[0]):
            pairs = [(self.labels[j], float(output[i, j])) for j in range(len(self.labels))]
            pairs.sort(key=lambda x: x[1], reverse=True)
            results.append([{"label": l, "score": s} for l, s in pairs])
        return results

    def infer_from_mel(self, log_mel_batch):
        """
        Feed log mel spectrograms directly, bypassing the model's own frontend.
        """
        output_dict = self._forward_from_mel(log_mel_batch)
        return self._to_predictions(output_dict['clipwise_output'].cpu().numpy())

    def infer_and_embed_from_mel(self, log_mel_batch):
        """
        Single forward pass returning (predictions, (L, 2048) embeddings).
        """
        output_dict = self._forward_from_mel(log_mel_batch)
        preds = self._to_predictions(output_dict['clipwise_output'].cpu().numpy())
        embs = output_dict['embedding'].cpu().numpy()
        return preds, embs

    def infer_all_layers_from_mel(self, log_mel_batch):
        """
        Single forward pass returning (predictions, dict of per-layer (L, C) embeddings).
        """
        model = self.model
        if hasattr(model, 'module'):
            model = model.module
        L = log_mel_batch.shape[0]

        injection = tt.tensor(log_mel_batch, dtype=tt.float32).to(device)
        injection = injection.unsqueeze(1).permute(0, 1, 3, 2)  # (L, 1, T, 64)

        layer_activations = {}
        handles = []

        handles.append(model.logmel_extractor.register_forward_hook(
            lambda module, inp, out: injection
        ))

        block_names = ['conv_block1', 'conv_block2', 'conv_block3', 'conv_block4', 'conv_block5', 'conv_block6']

        def make_hook(key):
            def hook(module, inp, out):
                assert out.ndim == 4, f"Expected 4D tensor from {key}, got {out.ndim}D"
                layer_activations[key] = out.detach().mean(dim=(-2, -1)).cpu().numpy()
            return hook

        for i, block_name in enumerate(block_names):
            handles.append(getattr(model, block_name).register_forward_hook(make_hook(f'layer_{i}')))

        try:
            dummy = tt.zeros(L, 32000, dtype=tt.float32).to(device)
            with tt.no_grad():
                output_dict = model(dummy, None)
        finally:
            for h in handles:
                h.remove()

        layer_activations['penultimate'] = output_dict['embedding'].detach().cpu().numpy()
        layer_activations['logits']      = output_dict['clipwise_output'].detach().cpu().numpy()  # (L, 527)
        preds = self._to_predictions(layer_activations['logits'])
        return preds, layer_activations


model = PannsModel()
