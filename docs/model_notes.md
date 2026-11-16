# The model: LeVJEPA-VideoMix-Large

Checkpoint: [galilai-group/LeVJEPA-VideoMix-Large](https://huggingface.co/galilai-group/LeVJEPA-VideoMix-Large)
Paper: [LeVJEPA: Efficient & Scalable Video Pretraining without the Heuristics](https://arxiv.org/abs/2608.27395)

This page is about three things: what the model is, how it was trained, and
what that tells us about reading it out for action recognition.

## 1. The checkpoint

| | |
|---|---|
| Architecture | ViT-L/16: 24 blocks, 16 heads, width 1024, 303M parameters |
| Input | `(B, 3, 16, 224, 224)`, ImageNet-normalised, frames at about 7.5 fps (one clip is about 2 s) |
| Tokens | tubelet size 1, so one token per 16x16 patch per frame: 16 x 14 x 14 = 3136 patch tokens, plus 1 CLS |
| Output | `last_hidden_state` `(B, 3137, 1024)`, CLS first, patches in `(t, h, w)` order; `pooler_output` `(B, 1024)` is the CLS |
| Positions | 3D rotary embeddings (time, height, width), no absolute position embedding |
| Attention | block-causal (see below) |
| Weights | the Polyak (EMA) average of the encoder; projector not included |
| Pretraining data | VideoMix: about 1.8M clips from Kinetics-710, Something-Something v2, Walking Tours and PE-Video. No labels. |
| Licence | CC-BY-NC-4.0: research and coursework only |

SSv2 is in the (unlabelled) pretraining data. That's fine for frozen probing,
V-JEPA does the same, but the report has to say it.

## 2. How LeVJEPA is trained

One encoder, one loss, no target network, no predictor, no pixel decoder.

1. **Views.** From a 16-frame clip, make one *global view* (full 224 px frame)
   and V *local views* (small random crops, 96 px, with colour/photometric
   augmentation). The released model uses V = 10. All views cover the same 16
   frames: they only differ in space and colour, never in time.
2. **Encoder.** Every view goes through the *same* encoder. A learnable CLS
   token is prepended to the patch tokens.
3. **Token dropping.** During training 95% of the patch tokens of each view are
   dropped at random right after the patch embedding. This makes training very
   cheap and, surprisingly, gives better features. At evaluation (`eval()`)
   nothing is dropped and the model sees all 3136 tokens.
4. **Projector.** Only the CLS output is used for the loss. A small MLP
   (1024 -> 2048 -> 256) maps it to an embedding `z`. The projector is thrown
   away after training.
5. **Loss** = invariance + λ · SIGReg, with λ = 0.02 (the only hyperparameter):
   - *Invariance*: the embedding of every local view should match the
     embedding of the global view of the same clip (mean squared error).
     Gradients go through both sides; there is no stop-gradient.
   - *SIGReg*: on its own, the invariance loss is solved by mapping every clip
     to the same vector (collapse). SIGReg prevents this by pushing the batch
     of embeddings towards an isotropic Gaussian. It projects the embeddings on
     1024 random directions and penalises how far each 1D projection is from a
     standard normal (Epps-Pulley test). A collapsed embedding has zero variance
     in some direction, so it is as far from a Gaussian as you can get.
6. **Block-causal attention.** Patch tokens attend to all patches of their own
   frame and of *earlier* frames, never later ones. The CLS token attends to
   everything, but no patch attends to the CLS.

## 3. What this means for the readout

This is the part that matters for our research question.

**The CLS token is the only token that was ever trained.** The loss reads only
the CLS (through the projector). It was trained to give the *same* answer for a
full frame and for small crops of it, so it is a clip-level summary that is
deliberately insensitive to *where* things are and to colour changes. It sees
all 16 frames.

**Patch tokens got no direct training signal**, yet they are far from useless.
The paper (Figures 3 and 5) shows that they group by object and that
similarity to one patch picks out the same object. We see the same on UCF101
clips: the person stands out from the background.

**Block-causal attention makes patch tokens depend on time.** The tokens of
frame t have only seen frames 0..t. Frame 0's tokens have seen one frame;
frame 15's tokens have seen the whole clip. Consequences:

- The features drift with t. If you do a PCA over all 16 frames at once, the
  first components mostly encode *which frame* a token is in, not what it shows.
- Mean-pooling all 3136 patch tokens mixes well-informed late tokens with
  early tokens that know nothing about motion.
- The 16 per-frame mean vectors (`frame_mean` in our cached features) are
  therefore not 16 independent frames: frame t's vector summarises frames 0..t.
  A temporal readout should take that into account.

**How the paper itself reads the model out:**

- An *attentive probe* (V-JEPA protocol): one learnable query cross-attends to
  all output tokens, then an MLP and a linear classifier. This is how all
  ImageNet and SSv2 numbers in the paper are obtained.
- A *linear probe on the average of all output tokens*, for Kinetics-400.

A linear probe on the CLS alone is not reported. Whether the trained CLS beats
pooled patch tokens, and how much an attentive or temporal readout adds, is
exactly what we want to find out.

## 4. Using it in this repo

```python
from src.models.backbone import LeVJEPABackbone
from src.utils.config import load_config

backbone = LeVJEPABackbone.from_config(load_config("configs/model.yaml"))
out = backbone(video)       # video (B, T, C, H, W), normalised
out.cls                     # (B, 1024)
out.patches                 # (B, 16, 14, 14, 1024)
```

- `src/data/video.py` does the preprocessing (7.5 fps sampling, 224 crop,
  ImageNet normalisation).
- `src/analysis/attention.py` gets attention maps out of any block. The model
  normally uses fused attention (SDPA), which never builds the attention
  matrix; we switch it off for one block at a time.
- Don't change `attn_mode`. Full attention runs without any error and silently
  gives worse features.
- Because of the block-causal mask, PyTorch can't use its fastest (flash)
  attention kernel, so expect more GPU memory than a standard ViT-L.
- Rough cost: about 15 s per clip on a laptop CPU; on a GPU it's a fraction of
  a second.
