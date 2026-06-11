# Breakthrough Research Directions

This document lists stronger research directions that could grow out of the
current ZK + neural-network robustness idea. The goal is to move beyond a simple
system combination and toward contributions that could support a serious
research paper.

---

## 1. ZK-Checkable Robustness Certificates

### Core Idea

Instead of proving full neural-network inference inside a zero-knowledge circuit,
the prover generates a robustness certificate outside the circuit and only proves
that the certificate is valid.

For example, for an input image `x` and perturbation radius `epsilon`, the prover
would show:

```text
For every x' within epsilon of x,
the model predicts the same class.
```

This can be certified using methods such as DeepPoly, CROWN, alpha-beta-CROWN, or
LP-based relaxations.

### Why This Could Be Important

Most ZKML systems prove exact inference:

```text
model(x) = y
```

But robustness requires a stronger claim:

```text
for all x' near x, model(x') = y
```

Proving this naively is very expensive. A ZK-checkable certificate could make
private robustness auditing practical.

### Possible Contribution

Design a certificate format that is:

- sound for neural-network robustness,
- cheap to verify inside ZK,
- compatible with private model weights and private inputs,
- usable for dataset-level certified accuracy.

### Main Challenge

The checker must still verify inequalities, ReLU cases, affine bounds, and
fixed-point arithmetic. If this ends up almost as expensive as normal ZK
inference, the idea loses its value.

---

## 2. Sublinear Certificate Checking

### Core Idea

Current certificate checking may still scale with the full model size:

```text
cost ~= number_of_weights * number_of_samples
```

That is too expensive for large neural networks.

The breakthrough direction is to make certificate checking compressed or
sublinear using cryptographic techniques such as:

- polynomial commitments,
- random linear checks,
- sumcheck protocols,
- folding schemes,
- vector commitments,
- sparse certificate representations.

### Why This Could Be Important

If we only move DeepPoly checking into ZK directly, the verifier may still need
to check a huge number of linear constraints. A sublinear checker would attack
the real bottleneck: dense affine layers.

The ideal result would be something closer to:

```text
cost << number_of_weights * number_of_samples
```

while still proving that the certificate corresponds to the committed model and
committed test data.

### Possible Contribution

Create a compressed proof system for neural-network robustness certificates where
the circuit does not explicitly check every weight-bound multiplication.

For example:

```text
Instead of checking all layer equations directly,
randomly combine them into a small number of algebraic checks.
```

### Main Challenge

Randomized checking must remain sound in zero knowledge. The prover should not be
able to hide incorrect bounds inside the compression. The method must also work
with private weights, private inputs, and finite-field arithmetic.

---

## 3. Fixed-Point Soundness For Neural Robustness In ZK

### Core Idea

DeepPoly and related certifiers are usually described using real-number
arithmetic, but ZK circuits operate over finite fields. This creates a gap:

```text
Real arithmetic certificate != finite-field ZK certificate
```

A publishable direction is to build a sound fixed-point arithmetic framework for
ZK robustness certificates.

### Why This Could Be Important

If rounding, overflow, or signed comparisons are handled incorrectly, the ZK
circuit may accept an invalid robustness certificate.

For safety-critical ML, this is a serious problem. A proof system that claims
robustness must be mathematically sound all the way down to the arithmetic used
inside the circuit.

### Possible Contribution

Define and implement:

- signed fixed-point encoding,
- safe rounding rules,
- overflow bounds,
- comparison and range-check gadgets,
- ReLU case validation,
- a theorem showing that accepted ZK certificates imply real robustness.

### Main Challenge

This direction is less flashy than a new protocol, but it is foundational. The
main difficulty is proving that the finite-field circuit exactly preserves the
intended robustness guarantee, including all approximation and rounding errors.

---

## Strongest Combined Paper Direction

The most promising research paper could combine all three directions:

```text
Sound and compressed zero-knowledge robustness certificates
for private certified-accuracy auditing.
```

The paper would aim to show:

1. robustness certificates can be checked in ZK,
2. the checking is sound under fixed-point finite-field arithmetic,
3. the checking is significantly cheaper than proving full ZK inference,
4. many per-image checks can be aggregated into one private certified-accuracy
   proof.

This would be stronger than only saying:

```text
We combine DeepPoly, ZK, and Nova.
```

The real contribution would be a reusable certificate-checking method for
privacy-preserving neural-network robustness.
