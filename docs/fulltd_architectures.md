# Fixed mean networks and objective implementations

The application code in `fulltd/models.py` implements the operations of the
following established architectures, with the compact dimensions and omissions
fixed by the study protocol. It is not an exact reproduction of reference
training scripts or their benchmark scores. Source revisions inspected on
2026-10-29 are recorded here; upstream license notices are retained in `licenses/`.

| Architecture | Reference revision | Mean parameters |
|---|---|---:|
| Electricity TCN | [locuslab/TCN](https://github.com/locuslab/TCN/tree/2f8c2b817050206397458dfd1f5a25ce8a32fe65) | 137,816 |
| FD001 TCN | same TCN reference | 92,801 |
| Electricity PatchTST | [yuqinie98/PatchTST](https://github.com/yuqinie98/PatchTST/tree/204c21efe0b39603ad6e2ca640ef5896646ab1a9) | 100,056 |
| METR-LA Graph WaveNet | [nnzhan/Graph-WaveNet](https://github.com/nnzhan/Graph-WaveNet/tree/6b162e80c59a1d494809252eca055cff93dc66b1) | 134,200 |
| Electricity DLinear | [cure-lab/LTSF-Linear](https://github.com/cure-lab/LTSF-Linear/tree/0c113668a3b88c4c4ee586b8c5ec3e539c4de5a6) | 8,112 |

TCN uses two causal kernel-3 convolutions and ReLU per residual block, plus a
ReLU after the residual addition. Electricity has six 64-channel blocks with
dilations 1,2,4,8,16,32; FD001 has four with dilations 1,2,4,8. A 1x1 residual
projection is present only when input and output channel counts differ. The
last temporal representation feeds a linear 24-output or scalar head. Neither
dropout nor weight/batch normalization is present.

PatchTST shares one univariate predictor across customer windows. There are
20 patches (length 16, stride 8, no end padding), width 64, two independently
initialized encoder layers, four attention heads, and a 128-wide GELU feedforward
sublayer. It retains the reference's **residual attention**: previous-layer
pre-softmax attention scores are added to current-layer scores, with gradients
through both. Residual sublayers use **post-LayerNorm**, matching the reference's
default `pre_norm=False` with the study's LayerNorm substitution. Flattening
uses width-then-patch order. Learned positional embeddings start uniformly in
[-0.02,0.02]. There is no RevIN, decomposition, pretraining, or dropout.

Graph WaveNet uses three supports: row-normalized supplied adjacency, its
separately row-normalized transpose, and row-softmax(ReLU(UV)) with rank 10.
Graph diffusion multiplies node features on the right by each support and its
second power, as in the reference. Eight gated temporal layers use dilations
1,2 repeated four times, kernel 2, and 32 residual/dilation channels. Skip and
end widths are 96 and 192. The receptive field is 13, so the 12-step history
receives one zero step on the left. A nodewise linear output head is exactly a
1x1 output convolution. The reference's obsolete Conv1d calls on 4-D tensors
are expressed as Conv2d operations. BatchNorm/dropout are absent. The reference
allocates unused residual 1x1 modules when graph convolution is active; this
implementation omits those dead parameters (8,448). The final layer's diffusion
output is also unused because the final skip is read before it; its dead
projection (7,200 parameters) is omitted as well. Together these explain the
difference from the 149,848 registered parameters in a literal compact port.
The point-prediction function is unchanged and every registered parameter
participates in prediction. Adaptive node embeddings use independent
standard-normal draws.

DLinear uses endpoint-replicated moving averages with fixed kernel 25. Two
shared 168-to-24 temporal projections predict seasonal and trend components
and their outputs sum. There are no customer-specific parameters.

Initialization is explicit: Kaiming-uniform for TCN convolutions and graph
pointwise convolutions, Xavier-uniform for linear/attention projections and
graph tanh/sigmoid gate convolutions, all biases zero, and LayerNorm scales one.
Stable SHA-256-derived streams separate mean initialization from graph adaptive
embeddings. The initialized mean depends only on dataset, backbone, and seed,
never the loss, rate, source sequence, or batch size. Gaussian variance heads
are zero-weight linear maps of the final hidden representation with bias
`inverse_softplus(1-1e-6)`, giving variance one initially. Their construction
does not alter the mean-network RNG stream.

Each algorithm has its own file under `agents/fulltd_*.py`. TD-full and
TD-scalar share only the exact residual-loss implementation: unrestricted
coefficients initialized at zero, full gradients through both endpoint means,
and a half-squared scalar loss averaged using the actual valid-pair count.
Type indices are zero-based 0/1 internally. The traffic intersection mask
qualifies the equality of zero-operator TD and ordinary MSE: equality holds on
complete data or the same explicitly chosen scalar support, while the ordinary
baseline retains every valid source label.
