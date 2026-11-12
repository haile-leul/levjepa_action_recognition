# Reading list

Core papers are split so that each of you presents one, but everyone reads the
abstract, figures and conclusion of all of them.

| When | Paper | Why |
|---|---|---|
| Sprint 1 | [V-JEPA: Revisiting Feature Prediction for Learning Visual Representations from Video](https://arxiv.org/abs/2404.08471) (Bardes et al., 2024) | Predict in latent space instead of pixels; frozen evaluation with probes |
| Sprint 1 | [LeJEPA: Provable and Scalable SSL Without the Heuristics](https://arxiv.org/abs/2511.08544) (Balestriero & LeCun, 2025) | Where SIGReg comes from and why it prevents collapse (sections 1, 4, 5) |
| Sprint 1 | [LeVJEPA: Efficient & Scalable Video Pretraining without the Heuristics](https://arxiv.org/abs/2608.27395) (Kuhn et al., 2026) | The model we use: one encoder, global and local views, token dropping |
| Sprint 2 | [VideoMAE](https://arxiv.org/abs/2203.12602) (Tong et al., 2022) | The masked-reconstruction alternative to JEPA |
| Sprint 2 | [V-JEPA 2](https://arxiv.org/abs/2506.09985) (Assran et al., 2025), evaluation and probe sections only | The attentive probe, our main "expressive readout" reference |
| Sprint 2 | [The "Something Something" video database](https://arxiv.org/abs/1706.04261) (Goyal et al., 2017) | Why some actions need motion, not just appearance |
| Optional | [ViT](https://arxiv.org/abs/2010.11929) (Dosovitskiy et al., 2020) | CLS token and patch tokens |
| Optional | [I-JEPA](https://arxiv.org/abs/2301.08243) (Assran et al., 2023) | JEPA on images |

Other links:

- Model card: [LeVJEPA-VideoMix-Large](https://huggingface.co/galilai-group/levjepa-videomix-large)
- SIGReg reference code: [rbalestr-lab/lejepa](https://github.com/rbalestr-lab/lejepa)

Keep your reading notes in `docs/notes/` (one markdown file per paper is fine).
They will save you a lot of time when writing the related work section.
