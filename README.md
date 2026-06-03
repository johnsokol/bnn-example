# A Pulsed Neural Network on MNIST — add-and-fire-on-overflow

see Blog post https://johnsokol.blogspot.com/2026/06/there-are-far-more-efficient-ways-to.html

A small, self-contained **pulsed (spiking) neural network** that classifies MNIST
**without a single multiplication at inference.** It is a working illustration of
a simple claim: running a neural network does not require dense floating-point
matrix multiplication. That is an assumption the industry standardized on — not a
law of computing.

This is **not** a ±1 / XNOR "binary" network. It is a *pulsed* network in the
spirit of **Palmo** pulse-stream signal processing: a value is carried by a
**stream of pulses** (magnitude = pulse rate), and polarity is carried by a
**sign clock** (here, the sign of a weight decides whether an arriving pulse adds
to or subtracts from a neuron's accumulator). The neuron is **integrate-and-fire**:

```
for each pulse arriving through weight w:
    accumulator += w               # the sign of w is the "sign clock"
    if accumulator overflows the threshold:
        fire (emit an output pulse)
        accumulator -= threshold   # the overflow carries over
```

**The overflow is the activation.** There is no compare-heavy "dot product then
threshold," and no multiply: a value×weight product is replaced by *delivering
the input's pulses, each adding the weight*, and the sum-then-threshold is
replaced by *accumulating until the accumulator overflows and fires*.

## Run it

```bash
python3 pulsed_nn.py
```

Only dependency is `numpy`.

### The MNIST data (`mnist.npz`)

You normally don't need to do anything — the script fetches the data for you.
On first use it looks for `mnist.npz` in this order:

1. `./mnist.npz` (next to the script)
2. `~/.keras/datasets/mnist.npz` (the standard Keras cache, if you have one)

If neither exists, it downloads `mnist.npz` once (~11 MB) to `./mnist.npz` from
the official Keras mirror:

```
https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
```

To grab it manually instead (e.g. for an offline machine):

```bash
curl -L -o mnist.npz \
  https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
```

`mnist.npz` is a standard NumPy archive with arrays `x_train (60000,28,28)`,
`y_train`, `x_test (10000,28,28)`, `y_test`. It is listed in `.gitignore` and is
**not** committed — it's a redownloadable dataset, not source — so cloning the
repo and running `python3 pulsed_nn.py` will fetch it automatically.

## What it produced (real run)

```
MNIST: 20000 train / 10000 test, 784 inputs, 10 classes

Training weights conventionally (offline) 784->128->10 ReLU ...
  trained in 28.5s; reference (floating-point) test accuracy: 94.23%

Pulsed integrate-and-fire inference (overflow = activation), first 1000 test images:
  pulses/input (T)   test accuracy
           1            11.40%
           2            58.50%
           4            88.20%
           8            93.00%
          16            93.70%
          64            93.60%
```

The pulsed network climbs from chance to within ~0.5% of the floating-point
reference as the **pulse budget T** grows, and saturates by about T = 16. That
curve *is* rate coding: more pulses = more precision = more energy. It is the
honest, fundamental knob of pulsed computation.

> The defaults (`TRAIN_SUBSET=20000`, `N_TEST=1000`) keep the script runnable on
> machines without an optimized BLAS. With normal numpy, set `TRAIN_SUBSET=0`
> (all 60k) and `N_TEST=10000` for full-scale numbers.

## Honest scope (what this does and does not show)

- **It does show:** the network's function is reproduced by pulse streams and
  overflow-firing, on a real dataset, with **no multiplications** in the
  inference path — only pulse-gated additions and threshold overflow.
- **It does not show:** a speedup in numpy on a CPU. A general-purpose CPU/GPU is
  built to maximize floating-point matrix-multiply throughput; emulating pulses on
  it is *slower*, not faster. The efficiency win of this style of computation is
  realized on hardware where pulses and accumulate-and-fire are cheap and floating
  point is absent — asynchronous many-core arrays (e.g. the **GreenArrays GA144**),
  FPGAs, or analog pulse hardware in the original Palmo spirit. Training is done
  conventionally and offline; the point here is the *inference substrate*.

## Why it matters

Most of the energy in modern inference goes into floating-point multiply-accumulates
and into moving 32-bit weights from memory. A pulsed, event-driven substrate
removes the multiplier, replaces values with pulse rates, and (on the right
hardware) does work only when a pulse actually arrives — energy proportional to
activity, not to a clock. This example is the minimal, runnable proof that the
arithmetic is optional; the larger argument for *post-matrix-multiply* hardware is
in the accompanying blog post.

## License

MIT — see [LICENSE](LICENSE). Copyright (c) 2026 John L. Sokol.
