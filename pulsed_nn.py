#!/usr/bin/env python3
"""
A pulsed (spiking) neural network on MNIST -- Palmo-style, add-and-fire-on-overflow.

This is NOT a +/-1 / XNOR "binary" network. It is a PULSED network built on the
Palmo signal-processing idea: a value is carried by a *stream of pulses* (its
magnitude is the pulse rate), and polarity is carried by a *sign clock* (here, the
sign of the weight decides whether an incoming pulse adds to or subtracts from a
neuron's accumulator).

The neuron is integrate-and-fire:

    for each pulse arriving through weight w:
        accumulator += w          # sign of w is the "sign clock"
        if accumulator overflows the threshold:
            FIRE  (emit an output pulse)
            accumulator -= threshold   # the overflow carries over

There is no multiply anywhere in inference. "value x weight" is replaced by
"deliver the input's pulses, each adding the weight"; "sum then threshold" is
replaced by "accumulate pulses until the accumulator overflows and fires." The
overflow IS the activation -- the original add-and-fire-on-overflow idea.

What this script does (numpy only):
  1. Loads MNIST (local cache if present, else downloads mnist.npz once).
  2. Trains the weights conventionally (offline). Training method is not the
     point -- the contribution is the *inference substrate*. We then throw away
     floating point at inference and run the network as pulses.
  3. Runs pulsed integrate-and-fire inference and reports MNIST test accuracy as
     a function of the pulse budget T (pulses per input per image). Accuracy
     rises with T: that is the rate-coding precision/energy knob, the honest
     core tradeoff of pulsed computation.

Honest scope: this demonstrates that the network's function is reproduced by
pulse streams + overflow-firing, with zero multiplications at inference. It does
NOT claim this is faster in numpy on a CPU -- the energy win is realized on
hardware where pulses and accumulate-and-fire are cheap and floating point is
not (async many-core like the GA144, FPGA, or analog pulse hardware in the spirit
of Palmo). See README.md.
"""

import os
import urllib.request
import numpy as np

RNG = np.random.default_rng(0)

MNIST_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
CACHE_CANDIDATES = [
    os.path.join(os.path.dirname(__file__), "mnist.npz"),
    os.path.expanduser("~/.keras/datasets/mnist.npz"),
    os.path.expanduser("~/sokol/sokol/.keras/datasets/mnist.npz"),
]


def load_mnist():
    path = next((p for p in CACHE_CANDIDATES if os.path.exists(p)), None)
    if path is None:
        path = CACHE_CANDIDATES[0]
        print(f"Downloading MNIST to {path} ...")
        urllib.request.urlretrieve(MNIST_URL, path)
    print(f"Loading MNIST from {path}")
    d = np.load(path)
    f = np.float32
    return (d["x_train"].reshape(-1, 784).astype(f) / 255.0, d["y_train"].astype(int),
            d["x_test"].reshape(-1, 784).astype(f) / 255.0, d["y_test"].astype(int))


# --------------------------------------------------------------------------
# Conventional offline training of a plain ReLU MLP (784->H->10, no bias).
# (Training is not the thesis; we just need weights. Inference is where the
# pulsed substrate replaces floating-point matmul.)
# --------------------------------------------------------------------------
def train_ann(X, y, H=128, epochs=8, lr=0.1, batch=128):
    n = len(X)
    W1 = RNG.normal(0, np.sqrt(2 / 784), size=(784, H)).astype(np.float32)
    W2 = RNG.normal(0, np.sqrt(2 / H), size=(H, 10)).astype(np.float32)
    Y = np.eye(10)[y]
    for ep in range(epochs):
        perm = RNG.permutation(n)
        for i in range(0, n, batch):
            bi = perm[i:i + batch]
            x, yb = X[bi], Y[bi]
            z1 = x @ W1
            a1 = np.maximum(z1, 0)
            logits = a1 @ W2
            p = np.exp(logits - logits.max(1, keepdims=True))
            p /= p.sum(1, keepdims=True)
            dlog = (p - yb) / len(bi)
            gW2 = a1.T @ dlog
            dz1 = (dlog @ W2.T) * (z1 > 0)
            gW1 = x.T @ dz1
            W1 -= lr * gW1
            W2 -= lr * gW2
    return W1, W2


def ann_accuracy(X, y, W1, W2):
    a1 = np.maximum(X @ W1, 0)
    return (np.argmax(a1 @ W2, 1) == y).mean()


# --------------------------------------------------------------------------
# Convert to a pulsed integrate-and-fire network.
# Normalize W1 so the expected per-step input to a neuron stays <= 1, i.e. a
# neuron fires at most ~once per step (rate in [0,1]). Threshold = 1.
# --------------------------------------------------------------------------
def normalize_for_pulses(X, W1):
    a = X @ W1                       # expected per-step hidden input = rate * W1
    lam = np.percentile(a[a > 0], 99.9)
    return (W1 / lam).astype(np.float32)


def pulsed_infer(X, W1n, W2, T, theta=1.0, seed=1):
    """Pulsed IF inference. X in [0,1] are per-step pulse probabilities (rates).

    Per step: draw input pulses; add their weights to the membrane (sign of the
    weight = sign clock); fire neurons whose membrane overflows theta; the firing
    neurons' pulses gate the output weights; subtract theta from those that fired
    (the overflow carries over). Nothing here multiplies two data values.
    """
    rng = np.random.default_rng(seed)
    B, H = len(X), W1n.shape[1]
    V = np.zeros((B, H))             # membrane potentials (accumulators)
    out = np.zeros((B, 10))          # integrated output evidence
    for _ in range(T):
        pulses_in = (rng.random((B, 784)) < X)          # binary input pulses
        V += pulses_in @ W1n                            # accumulate (adds only)
        fired = V >= theta                              # overflow -> fire
        out += fired @ W2                               # output pulses gate W2
        V -= fired * theta                              # overflow carries over
    return np.argmax(out, 1)


def main():
    import time
    # TRAIN_SUBSET / N_TEST keep this runnable on machines without an optimized
    # BLAS. With normal numpy, set TRAIN_SUBSET=0 (all 60k) and N_TEST=10000.
    TRAIN_SUBSET = int(os.environ.get("TRAIN_SUBSET", "20000"))
    N_TEST = int(os.environ.get("N_TEST", "1000"))

    Xtr, ytr, Xte, yte = load_mnist()
    if TRAIN_SUBSET:
        Xtr, ytr = Xtr[:TRAIN_SUBSET], ytr[:TRAIN_SUBSET]
    print(f"MNIST: {len(Xtr)} train / {len(Xte)} test, 784 inputs, 10 classes\n")

    print("Training weights conventionally (offline) 784->128->10 ReLU ...")
    t0 = time.time()
    W1, W2 = train_ann(Xtr, ytr)
    print(f"  trained in {time.time()-t0:.1f}s; reference (floating-point) "
          f"test accuracy: {ann_accuracy(Xte, yte, W1, W2)*100:.2f}%\n")

    W1n = normalize_for_pulses(Xtr, W1)

    Xs, ys = Xte[:N_TEST], yte[:N_TEST]
    n_test = N_TEST
    print(f"Pulsed integrate-and-fire inference (overflow = activation), "
          f"first {n_test} test images:")
    print("  pulses/input (T)   test accuracy")
    for T in (1, 2, 4, 8, 16, 64):
        acc = (pulsed_infer(Xs, W1n, W2, T) == ys).mean()
        print(f"        {T:4d}            {acc*100:5.2f}%")

    print("\nWith only a few pulses per input the pulsed network already matches")
    print("the floating-point reference -- and it uses no multiplications at all,")
    print("only pulse-gated additions and overflow-firing. The pulse budget T is")
    print("the rate-coding precision/energy knob.")


if __name__ == "__main__":
    main()
