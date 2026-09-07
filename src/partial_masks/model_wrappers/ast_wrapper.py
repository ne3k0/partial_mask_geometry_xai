# AST (Audio Spectrogram Transformer) fine-tuned on AudioSet with ViT pretraining
import platform

import torch as tt
import numpy as np
import transformers
from transformers import pipeline

if platform.uname().system == "Darwin":
    device = tt.device("mps")
else:
    device = tt.device("cuda")

model_id = "MIT/ast-finetuned-audioset-10-10-0.4593"


class Model:
    def __init__(self, model_id) -> None:
        self.model_id = model_id
        self.pipe = pipeline("audio-classification", model=model_id, device=device, top_k=None)

    def _prepare_input(self, log_mel_batch):
        """
        Transpose, pad/truncate to AST's fixed length, apply AST normalisation.
        """
        fe = self.pipe.feature_extractor
        max_length = fe.max_length  # 1024

        x = log_mel_batch.transpose(0, 2, 1)  # (L, T, n_mels)
        T = x.shape[1]
        if T < max_length:
            x = np.pad(x, ((0, 0), (0, max_length - T), (0, 0)))
        elif T > max_length:
            x = x[:, :max_length, :]

        if hasattr(fe, 'mean') and fe.mean is not None:
            x = (x - fe.mean) / (fe.std * 2)

        return tt.tensor(x, dtype=tt.float32).to(device)

    def _to_predictions(self, probs):
        id2label = self.pipe.model.config.id2label
        results = []
        for i in range(probs.shape[0]):
            pairs = [(id2label[j], float(probs[i, j])) for j in range(len(id2label))]
            pairs.sort(key=lambda x: x[1], reverse=True)
            results.append([{"label": l, "score": s} for l, s in pairs])
        return results

    def infer_from_mel(self, log_mel_batch):
        """
        Feed log mel spectrograms directly to AST, bypassing AutoFeatureExtractor.
        log_mel_batch: numpy array shape (L, n_mels, T).
        """
        x = self._prepare_input(log_mel_batch)
        with tt.no_grad():
            output = self.pipe.model(input_values=x)
        probs = tt.sigmoid(output.logits).cpu().numpy()
        return self._to_predictions(probs)

    def infer_and_embed_from_mel(self, log_mel_batch):
        """
        Single forward pass returning (predictions, (L, 768) CLS embeddings).
        """
        x = self._prepare_input(log_mel_batch)
        with tt.no_grad():
            output = self.pipe.model(input_values=x, output_hidden_states=True)

        probs = tt.sigmoid(output.logits).cpu().numpy()  # multi-label, see infer_from_mel
        last_hidden = output.hidden_states[-1]                                # (L, seq, 768)
        embs = ((last_hidden[:, 0] + last_hidden[:, 1]) / 2).cpu().numpy()    # (L, 768)
        return self._to_predictions(probs), embs

    def infer_all_layers_from_mel(self, log_mel_batch):
        """
        Single forward pass returning (predictions, dict of per-layer (L, 768) embeddings).
        """
        x = self._prepare_input(log_mel_batch)
        with tt.no_grad():
            output = self.pipe.model(input_values=x, output_hidden_states=True)

        probs = tt.sigmoid(output.logits).cpu().numpy()
        n = len(output.hidden_states)
        layer_embs = {}
        for k, hs in enumerate(output.hidden_states):
            key = 'penultimate' if k == n - 1 else f'layer_{k}'
            layer_embs[key] = ((hs[:, 0] + hs[:, 1]) / 2).cpu().numpy()
        layer_embs['logits'] = probs  # (L, 527)
        return self._to_predictions(probs), layer_embs


model = Model(model_id)
