# Reading list

Each of you presents the papers you read at the meeting that closes the sprint.

| When | Paper | Why |
|---|---|---|
| Sprint 1 | [V-JEPA: Revisiting Feature Prediction for Learning Visual Representations from Video](https://arxiv.org/abs/2404.08471) (Bardes et al., 2024) | Predict in latent space instead of pixels; frozen evaluation with probes |
| Sprint 1 | [LeJEPA: Provable and Scalable SSL Without the Heuristics](https://arxiv.org/abs/2511.08544) (Balestriero & LeCun, 2025) | Where SIGReg comes from and why it prevents collapse (sections 1, 4, 5) |
| Sprint 1 | [LeVJEPA: Efficient & Scalable Video Pretraining without the Heuristics](https://arxiv.org/abs/2608.27395) (Kuhn et al., 2026) | The model we use: one encoder, global and local views, token dropping |
| Sprint 2 | [UCF101: A Dataset of 101 Human Actions Classes From Videos in The Wild](https://arxiv.org/abs/1212.0402) (Soomro et al., 2012) | Our first, smaller dataset |
| Sprint 2 | [The "Something Something" video database](https://arxiv.org/abs/1706.04261) (Goyal et al., 2017) | Our second dataset, where you need motion to tell actions apart |

Background, when you need it:

| Paper | Why |
|---|---|
| [V-JEPA 2](https://arxiv.org/abs/2506.09985) (Assran et al., 2025), evaluation section | The attentive probe, reference for attention-based readouts |
| [VideoMAE](https://arxiv.org/abs/2203.12602) (Tong et al., 2022) | The masked-reconstruction alternative to JEPA |
| [ViT](https://arxiv.org/abs/2010.11929) (Dosovitskiy et al., 2020) | CLS token and patch tokens |
| [I-JEPA](https://arxiv.org/abs/2301.08243) (Assran et al., 2023) | JEPA on images |

Other links:

- Model card: [LeVJEPA-VideoMix-Large](https://huggingface.co/galilai-group/LeVJEPA-VideoMix-Large)
- SIGReg reference code: [rbalestr-lab/lejepa](https://github.com/rbalestr-lab/lejepa)
- What we know about the model and its outputs: [model_notes.md](model_notes.md)
