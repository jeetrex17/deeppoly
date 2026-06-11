# Research Idea: ZK Proof of Certified Accuracy via Abstract Interpretation

> A novel system that proves a neural network achieves ≥ X% certified robustness
> on a private test set — without revealing model weights or test data.

---

## The Problem

When a company deploys a safety-critical ML model (medical diagnosis, fraud detection,
autonomous driving), two questions arise that cannot both be answered today:

1. **Regulators ask:** "Prove your model is robust to adversarial inputs."
2. **Companies respond:** "Only if you let us hide our weights and test data."

There is no system that satisfies both simultaneously. Either the company reveals
proprietary model weights, or the regulator cannot verify the claim.

---

## Our Goal

Build a cryptographic system where a model owner can produce a single proof that says:

> *"My model achieves ≥ X% certified accuracy at perturbation radius ε on a private
> test set. You can verify this proof in under 1 second without ever seeing my model
> weights, test images, or labels."*

**Certified accuracy** is stronger than plain accuracy — it means for X% of test
images, no attacker within an ε-ball around the image can flip the prediction.
This is a formal, mathematical guarantee, not an empirical claim.

---

## Key Insight: LP Certificate Verification

The central bottleneck in all prior ZK-for-ML work is encoding ReLU activations
inside a ZK circuit. Exact ReLU requires bit decomposition — roughly 32 range
checks per neuron. A network with 1000 neurons needs ~32,000 constraints per image.

We sidestep this entirely using **LP certificate verification**:

```
Standard approach (expensive):
  ZK circuit must RECOMPUTE: run every ReLU inside the proof
  → 32,000 constraints per image

Our approach (cheap):
  1. Run DeepPoly OUTSIDE the circuit (offline, no speed limit)
  2. Save bounds [l_i, u_i] for every neuron
  3. ZK circuit only CHECKS: are these bounds consistent with the weights?
  → ~70 constraints per image
```

Checking consistency is just verifying linear inequalities — purely linear arithmetic,
no non-linear operations anywhere in the circuit.

### Why This Is Sound

Model weights are **committed** (cryptographically locked) at proof start.
The consistency check uses those committed weights. Fake bounds cannot pass
the check without breaking the commitment scheme — computationally infeasible.

### The Three Cases for ReLU (All Linear)

```
Case 1 — always active  (l ≥ 0):
    check: l_out = l_in,  u_out = u_in

Case 2 — always inactive  (u ≤ 0):
    check: l_out = 0,  u_out = 0

Case 3 — mixed  (l < 0 < u):
    check: u_out ≤ u/(u-l) · u_in      ← DeepPoly linear upper relaxation
    check: l_out ≥ 0                    ← DeepPoly linear lower bound
```

Sign checks are still needed to determine which case applies (is l ≥ 0? is u ≤ 0?).
What is eliminated is bit decomposition for the ReLU computation itself.
One sign check per neuron for case selection — not one per bit per neuron.
The saving is real but more modest than "no checks at all."

---

## Trustless Design — No Trusted Setup

The system is fully trustless. No ceremony, no trusted third party, no structured
reference string. Security rests on discrete logarithm hardness alone.

KZG polynomial commitments are explicitly rejected — they require a trusted setup
ceremony (like Zcash's 200-person MPC). Every component below requires no setup.

### Pedersen Vector Commitments — Lock the Weights

```
Commit to all weights w[0..n]:
  C = w[0]·G_0 + w[1]·G_1 + ... + w[n]·G_n

where G_0..G_n are random public curve generators (no secret τ needed)

Properties:
  Hiding:        C reveals nothing (discrete log hardness)
  Binding:       cannot open C to different weights
  Homomorphic:   C(w) + C(v) = C(w+v)   ← key for bound propagation
  Trusted setup: NONE
```

The homomorphic property is better than KZG for our use case — bound propagation
through linear layers is linear combinations of committed values, which Pedersen
handles natively without any circuit constraints.

### Bulletproof Inner Product Arguments — Linear Layers for Free

Each linear layer is matrix multiplication = a series of dot products.
Bulletproofs prove an inner product ⟨a, b⟩ = c in **O(log n) communication**
using only elliptic curve operations — no arithmetic circuit, no trusted setup:

```
Standard circuit:    n constraints per dot product
Bulletproof IPA:     O(log n) elliptic curve operations per dot product
Trusted setup:       NONE
```

Since LP certificate verification is all linear arithmetic, this eliminates
arithmetic circuits for the main computation entirely.

### What Still Needs a Range Check

Only the final certification condition:
```
lower_bound[true_class] > upper_bound[wrong_class]
```
This is a Bulletproof range proof — the exact same construction Monero uses
in production for transaction amount validity. No trusted setup.

### Full Trustless Architecture

```
Model weights → Pedersen vector commitment  (no trusted setup)
                      ↓
Per-image: LP certificate verification
           → Bulletproof IPA for linear layer bounds
           → Bulletproof range proof for certification condition
                      ↓
Nova folding to aggregate N images
(Nova uses Pedersen commitments internally — no trusted setup)
                      ↓
Final proof: constant size, < 1 second to verify
```

### Trust Assumption

```
The entire system is secure if:
  1. Discrete logarithm is hard on the chosen curve
  2. The hash function (for Merkle tree) is collision-resistant

Nothing else. No ceremony. No trusted party. No structured reference string.
```

### What the Trustless Upgrade Gains vs KZG

| Component | KZG approach | Trustless approach |
|---|---|---|
| Weight commitment | KZG (trusted setup required) | Pedersen (no setup) |
| Weight proof cost | O(1) pairing | O(log n) IPA |
| Linear layer proof | n constraints | O(log n) IPA |
| Final certification check | Range check in circuit | Bulletproof range proof |
| Security assumption | Pairing + discrete log + trusted setup | Discrete log only |
| Framework | Groth16 / PLONK | Halo2 / Nova |

---

## Full System Architecture

```
                    OFFLINE (no ZK constraints)
                    ┌──────────────────────────────┐
                    │  Run DeepPoly on each image  │
                    │  Save bounds [l_i, u_i]       │
                    │  for every neuron             │
                    └────────────┬─────────────────┘
                                 │
                    ONLINE (inside ZK proof)
                                 │
                    ┌────────────▼─────────────────┐
                    │  Per-image ZK circuit         │
                    │  (LP certificate verification)│
                    │                               │
                    │  1. Check bounds consistent   │
                    │     with committed weights    │
                    │  2. Check certification:      │
                    │     l[true] > u[wrong_class]  │
                    └────────────┬─────────────────┘
                                 │  × N images
                    ┌────────────▼─────────────────┐
                    │  Nova Folding                 │
                    │                               │
                    │  Aggregates N per-image       │
                    │  proofs into ONE proof        │
                    │  Proof size: constant in N    │
                    └────────────┬─────────────────┘
                                 │
                    ┌────────────▼─────────────────┐
                    │  Final Proof                  │
                    │                               │
                    │  "≥ X% of test images are    │
                    │   certified robust at ε"      │
                    │                               │
                    │  Verified in < 1 second       │
                    │  Proof size: < 10 KB          │
                    └──────────────────────────────┘
```

---

## Four Contributions

```
┌──────────────────────────────────────────────────────────┐
│   ZK proof of certified accuracy over test set           │  ← GOAL
│                                                          │
│   ┌──────────────────────────────────────────────────┐   │
│   │   DeepPoly + ZK + Nova folding combined          │   │  ← SYSTEM
│   │                                                  │   │
│   │   ┌──────────────────────────────────────────┐   │   │
│   │   │   LP certificate verification            │   │   │  ← TRICK
│   │   │   (DeepPoly abstract domain in ZK)       │   │   │
│   │   │                                          │   │   │
│   │   │   ┌──────────────────────────────────┐   │   │   │
│   │   │   │   Fixed-point soundness theorem  │   │   │   │  ← PROOF
│   │   │   │   (real arithmetic → ZK circuit) │   │   │   │
│   │   │   └──────────────────────────────────┘   │   │   │
│   │   └──────────────────────────────────────────┘   │   │
│   └──────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────┘
```

| Contribution | Type | Required for |
|---|---|---|
| LP certificate verification | Engineering | Workshop paper |
| Nova folding for test-set aggregation | Engineering | Conference paper |
| Fixed-point soundness theorem | Theory | Top venue (CCS/S&P) |
| End-to-end system + benchmarks | Empirical | All venues |

---

## What Makes This Novel

No prior paper has combined these three components:

| Component | Prior work exists? | Our use |
|---|---|---|
| DeepPoly certification | Yes (Singh et al., POPL 2019) | Offline — produces certificate |
| ZK proof of single inference | Yes (EZKL, zkLLM) | Not used — too expensive |
| ZK proof of fairness certificate | Yes (FairProof, ICML 2024) | Related — different certificate type |
| ZK proof of training | Yes (Kaizen, CCS 2024) | Related — different goal |
| Nova folding for ML | Partial | Applied to certification aggregation |
| **LP certificate verification for DeepPoly** | **No** | **Core contribution** |
| **ZK proof of certified accuracy over test set** | **No** | **Final result** |

### Key Differentiation from FairProof (Closest Paper)

FairProof (ICML 2024, UC San Diego + Stanford) is the most structurally similar work.
They use ZK to prove **individual fairness** of a model without revealing weights.

| | FairProof | Our Work |
|---|---|---|
| Certificate type | Local Lipschitz / sensitivity bound | DeepPoly polyhedral bounds |
| What is certified | Fairness (similar inputs → similar outputs) | Robustness (ε-ball → same class) |
| Scope | Single input | Full test set |
| Aggregation | None | Nova folding over N images |
| Output | One fairness certificate | Certified accuracy percentage |

---

## Inspiration: How Zcash and Monero Solved Fixed-Point Soundness

Before describing our theorem, it is worth understanding how the two most
successful deployed ZK systems handled the same problem — and stealing their solution.

### The Zcash / Monero Trick

The "decimal" amounts you see in Zcash (0.5 ZEC) and Monero (1.23 XMR) are
purely wallet display conventions. Inside every circuit, proof, and commitment
**only integers ever exist**:

```
Zcash:   1 ZEC  = 100,000,000 zatoshi  (10^8)
         Circuit sees: 50,000,000       (never 0.5)

Monero:  1 XMR  = 1,000,000,000,000 piconero  (10^12)
         Circuit sees: 1,230,000,000,000        (never 1.23)
```

They eliminated the fixed-point problem entirely by choosing the smallest
denomination before designing the circuit. The BLS12-381 scalar field is
~255 bits wide. Max ZEC supply fits in ~51 bits. Overflow is impossible by
construction. Monero's Bulletproofs then prove `amount ∈ [0, 2^64)` as a
pure integer with no rounding anywhere.

**Soundness is trivial for them because the circuit IS the real arithmetic.**

### How We Apply This to Neural Network Weights

We use the exact same pattern. Choose a scaling factor S upfront and quantize
everything before the circuit touches it:

```
Choose S = 2^16  (65,536)  — or S = 2^20 for higher precision

Weight  w =  0.7312  →  w_int =  47,923  (= floor(0.7312 × 65536))
Bound   l = -0.3000  →  l_int = -19,661
Bound   u =  0.8000  →  u_int =  52,429

Circuit sees only integers. No floating point. No rounding inside the circuit.
```

This is Zcash's atomic-unit trick applied to neural network weights.

### The Safe Rounding Direction — The Key Insight

One rounding decision remains: which direction do we round when converting
real bounds to integers? This is where soundness is won or lost.

```
Lower bounds:  always FLOOR  →  l_int  ≤  l_real × S
Upper bounds:  always CEIL   →  u_int  ≥  u_real × S

Effect: integer bounds are WIDER (more conservative) than real bounds.
        l_int/S ≤ l_real  (integer lower is smaller)
        u_int/S ≥ u_real  (integer upper is larger)
```

Wider integer bounds make the certification check HARDER to pass:
- l_int_true is smaller than l_real_true → harder to exceed u_int_wrong
- u_int_wrong is larger than u_real_wrong → harder to beat

```
Soundness argument:
  If l_int_true > u_int_wrong   (integer circuit certifies)
  Then floor(l_real_true × S) > ceil(u_real_wrong × S) ≥ u_real_wrong × S
  Therefore l_real_true × S > u_real_wrong × S
  Therefore l_real_true > u_real_wrong   (real robustness holds) ✓
```

The harder integer check passing guarantees the easier real check passes.
That is what makes soundness work — not "tighter" bounds but "conservative" ones.

---

## Fixed-Point Soundness — The Required Theorem

### Formal Statement

```
Theorem (Fixed-Point Soundness):

Let S = 2^k be the chosen scaling factor.
Let w_int = floor(w · S) for all weights w.
Let l_int = floor(l_real · S),  u_int = ceil(u_real · S)
for all DeepPoly bounds.

If the ZK circuit accepts the certificate
  {w_int, l_int_i, u_int_i for all neurons i}

Then the real-number DeepPoly bounds
  {w, l_real_i, u_real_i}
also satisfy the certification condition:

  lower_bound[true_class] > upper_bound[wrong_class]

And therefore: for all x' within ε of x,
  the model predicts the same class as x.
```

### What Needs to Be Proved

The theorem is doable but not simple. Six sub-problems must all be handled:

**1. Negative weights — rounding direction flips**
```
For a positive weight w > 0:
    floor(w × S) × floor(l × S)  ≤  w × l × S²    ← conservative ✓

For a negative weight w < 0:
    floor(w × S) × floor(l × S)  ≥  w × l × S²    ← direction FLIPS ✗

Fix: use floor for positive weights, ceil for negative weights.
     The rounding direction depends on the sign of each weight individually.
```

**2. Multiplication rounding accumulates**
```
Each weight × bound multiplication introduces error up to 1 unit.
A layer with n inputs accumulates up to n units of rounding error.
After L layers: accumulated error ≤ n^L units.

Requirement: choose S large enough that n^L / S < certification margin.
For MNIST (n=784, L=3): need S >> 784³ ≈ 5 × 10^8 → S = 2^30 safer than 2^16.
```

**3. Mixed ReLU slope rounding**
```
Real:     u_out ≤  u / (u - l)  ·  x

Integer:  need ceil(u_int / (u_int - l_int)) — ceiling of a ratio.
          Standard integer division truncates (floors), which LOOSENS the bound.

Fix:      ceil(a/b) = floor((a + b - 1) / b)  — one extra addition in the circuit.
```

**4. Overflow bounds**
```
After L layers of fixed-point arithmetic, values grow by factor S per layer.
After L layers: magnitude ≤ S^L × original_magnitude.

For BLS12-381 (255-bit field) and S = 2^30, L = 10:
    S^L = 2^300 > 2^255   ← overflow!

Fix: renormalize (divide by S) after each layer, OR choose smaller S,
     OR use a larger field. Document the exact constraint.
```

**5. Certification margin slack**
```
The certification condition is:
    l_int_true  >  u_int_wrong

After rounding, there may be a gap of several units between integer
and real certification. The margin must exceed the total rounding error.
If the real margin is tiny (near-boundary certified images), integer
rounding may flip a certificate from pass to fail.

This reduces completeness: some truly robust images may not be certifiable
in the integer system if their real margin < accumulated rounding error.
```

**6. Lean formalization covers all six cases**
The pen-and-paper proof handles these symbolically.
The Lean proof forces every case to be explicit and machine-checked.
This is why Lean matters — not because the math is deep but because
there are many small cases that are easy to miss.

### Why This Matters for the Paper

Without this theorem: engineering contribution → workshop paper.
With this theorem: formally verified cryptographic protocol → CCS / S&P.

Zcash never had to write this theorem because their arithmetic is exact.
We do — and that difference is what makes our paper novel even beyond the
engineering. It is the first formal soundness proof for ZK neural network
robustness certificates under finite-field arithmetic.

### Lean Formalization

The soundness theorem will be formally verified in **Lean 4** using Mathlib.

Why Lean:
- Machine-checked — no reviewer can dispute the proof
- Mathlib already contains interval arithmetic lemmas needed for our argument
- Growing standard in ZK systems (RISC Zero, Ethereum Foundation use Lean/Coq)
- Published as a citable artifact alongside the paper

Sketch of the Lean statement:

```lean
theorem fixed_point_soundness
    (S : ℕ) (hS : S = 2^16)
    (l_real u_real l_int u_int : ℤ)
    (hl : l_int = ⌊l_real * S⌋)
    (hu : u_int = ⌈u_real * S⌉)
    (h_circuit : l_int_true > u_int_wrong) :
    l_real_true > u_real_wrong := by
  -- proof by monotonicity of floor/ceil and safe rounding direction
  ...
```

### When to Tackle This

Phase 2: write the pen-and-paper proof as the argument becomes concrete.
Phase 3: formalize in Lean alongside the Nova folding implementation.
Phase 4: Lean proof as a published artifact — the main theorem of the paper.

---

## Formal Adversarial Model

### Parties

- **Prover P** — the model owner (company, hospital, government agency)
- **Verifier V** — the regulator, auditor, or user

### Information Partition

```
Private (P only):               Public (both parties):
  Model weights w                 Weight commitment    C_w
  Test images {x_1..x_N}         Test set Merkle root C_X
  Labels {y_1..y_N}              Label Merkle root    C_Y
  DeepPoly bounds {[l_i, u_i]}   Certified accuracy   k/N
                                  Perturbation radius  ε
                                  The proof            π
```

### Three Security Properties

**Soundness** — protects the verifier from a lying prover:
```
A cheating P* cannot produce a valid proof π for a false claim.

Formally:
  Pr[ V accepts π  AND  true certified accuracy < k/N ] ≤ negl(λ)

Breaks if: discrete logarithm is easy (no known attack)
```

**Zero-Knowledge** — protects the prover's IP from a curious verifier:
```
A cheating V* learns nothing beyond the certified accuracy k/N.
Cannot learn: individual weights, test images, labels,
              which specific images failed or passed certification.

Formally:
  ∃ simulator S such that
  View_{V*}(π) ≈_c S(C_w, C_X, C_Y, k/N, ε)

Breaks if: discrete logarithm is easy
```

**Completeness** — the system works for honest parties:
```
An honest P with a truly certified model always produces
a valid proof that an honest V accepts.

Formally: Pr[ V accepts π  |  true certified accuracy ≥ k/N ] = 1
```

### Adversary Capabilities

The adversary is computationally bounded (polynomial time) and may:
- See all public parameters before deciding to cheat
- Choose the false claim adaptively
- Rewind the verifier (standard ZK adversary model)
- Run any polynomial-time algorithm

### Explicit Non-Goals

The system does NOT protect against:
- **Test set cherry-picking** — the prover chooses what to commit to as the test set.
  Preventing this requires a separate certified test set protocol (out of scope).
- **Side-channel attacks** — hardware leakage during proof generation.
- **Prover-verifier collusion** — if both parties cooperate, no cryptographic
  system can help.
- **Quantum adversaries** — Bulletproofs and Pedersen commitments are not
  post-quantum. Post-quantum variant is future work (use lattice commitments).

### Trust Assumption (Formal)

```
The system is (soundness, ZK, completeness)-secure under:
  1. Hardness of discrete logarithm on the chosen elliptic curve
  2. Collision resistance of the Merkle hash function

No trusted setup. No structured reference string. No ceremony required.
```

---

## Three Versions of the System

The paper is structured in three increasing levels of privacy and difficulty.
Each version is independently publishable. Do not try to build Version C first.

---

### Version A — Public Weights, Private Test Data (Build This First)

Weights are committed and visible to the verifier. Only the test set is hidden.

**Why start here:** The hardest sub-problem (private matrix-vector multiplication)
disappears. The circuit only needs to:
1. Verify that DeepPoly bounds are consistent with the known weights
2. Hide which specific test images passed or failed
3. Output the certified accuracy count

**Constraint count for a 3-layer MNIST MLP:**

A 3-layer MNIST MLP has approximately 784×100 + 100×100 + 100×10 = 89,400 weights.

| Component | Constraints (public weights) |
|---|---|
| Linear layer verification (W known) | ~300 per layer |
| ReLU case selection (sign per neuron) | ~200 |
| Overflow / range validity | ~100 |
| Certification margin check | ~10 |
| **Total per image** | **~1,000** |

With public weights each layer check is cheap — verify `W·l_in ≤ l_out` using
known W. No private multiplication needed. Actual numbers will be measured
in Phase 2 benchmarking.

**What this proves:** "The test set was evaluated honestly and ≥ k/N images are
certified robust" — with the model architecture and weights visible to the verifier.

**Privacy model A:** Test data private. Model public (or available to regulator).

---

### Version B — Private Weights + Private Test Data + Nova Aggregation

Both model and test data are hidden. This is the full privacy goal.

**The true bottleneck:** Private matrix-vector multiplication.

```
prove:  y = W · x
where:  W is private (committed model weights)
        x is private (committed test image bounds)
        y is private (committed output bounds)
```

In R1CS each private multiplication w_ij × x_j costs one constraint.
For MNIST layer 1 alone: 784 × 100 = 78,400 constraints per image.

| Approach | Constraints per image |
|---|---|
| Exact ReLU, private weights (EZKL style) | ~300,000+ |
| LP cert verification, public weights (Version A) | ~1,000 |
| **LP cert verification, private weights (Version B)** | **~90,000** |

Still a meaningful reduction over exact-ReLU with private weights, because
we verify fewer operations (bounds only, not exact activations). But the
constraint count is dominated by private weight multiplications, not ReLU.

**Privacy model B:** Both test data and model weights private.

---

### Version C — Private Weights via DualMatrix (The Full System)

Replace the R1CS private matrix-vector multiplication with DualMatrix
(ePrint 2025/1768, Springer Cybersecurity 2026).

**DualMatrix proves Y = W·X where both W and X are privately committed:**
```
Prover time:   O(N + n) where N = non-zero elements in W
               O(n²) for dense, O(N) for sparse — exploits sparsity
Proof size:    O(log n)
Verifier time: O(log n)
Trusted setup: NONE — fully transparent
Technique:     Extends Bulletproofs using rank-1 property + two-tier
               Pedersen commitment scheme + bilinear group algebra
```

**What this gives:**

| Metric | Version B (R1CS) | Version C (DualMatrix) |
|---|---|---|
| Prover per layer (dense) | O(n²) constraints | O(n²) group ops |
| Prover per layer (sparse) | O(n²) constraints | **O(N) group ops** |
| Proof size | Linear in constraints | O(log n) per layer |
| Trusted setup | None | **None** |
| Status | Engineering | Apply existing tool |

For a 90% pruned model: DualMatrix prover is 10× cheaper than R1CS.
For a 99% pruned model: 100× cheaper.

**Why DualMatrix fits our trustless goal:**
DualMatrix uses a transparent setup — no ceremony, no trusted party.
This matches our security model exactly (discrete log hardness only).

**The integration challenge:**
DualMatrix produces its own proof. Nova folds a single relation.
These must be expressed as one unified foldable relation in Halo2.
This is the remaining engineering challenge of Phase 4.

**Privacy model C:** Both weights and test data private.
Prover work scales with actual non-zero weights, not matrix dimensions.

---

### Note on Weight Quantization (All Versions)

The system certifies the quantized model `w_int/S`, not the original
floating-point model `w`. The prover commits to and deploys `w_int/S`
as the actual model. This is not a weakness — quantized inference is
standard practice in production ML and reduces the gap between the
certified and deployed model.

---

## Implementation Plan

### Tools
- **Python** — run DeepPoly, save per-neuron bounds (already have this)
- **Gnark** (Go) — ZK circuit implementation (FairProof used this, reference code exists)
- **Nova-Scotia** — Nova folding wrapper for Gnark circuits
- **Bulletproofs** — inner product arguments for linear layers (optional upgrade)

### Phased Roadmap

Do NOT try to build everything at once. Each phase is independently publishable.

**Phase 1 — Read and understand (Month 1-2)**
Read FairProof end to end. Understand their Gnark circuit implementation.
This is your template. Also read the Nova paper and run their example code.
Output: deep understanding of the ZK engineering required.

**Phase 2 — Version A prototype (Month 3-4)**
Public weights, private test data. Implement LP certificate verification
in Gnark for a single MNIST image with known weights.
Measure actual constraint count (do not estimate — build and benchmark).
Pen-and-paper fixed-point soundness proof for the simple public-weight case.
Output: workshop paper. Version A alone is publishable.

**Phase 3 — Version B: full privacy + Nova (Month 5-6)**
Private weights via Pedersen commitments. Add Nova folding over N images.
Merkle tree for test set. Benchmark constraint count with private weights
(expect ~90,000/image for MNIST — document honestly).
Output: conference paper (SaTML / IEEE S&P workshop / USENIX Security workshop).

**Phase 4 — Version C: DualMatrix for private weights (Month 7-12)**
Integrate DualMatrix (ePrint 2025/1768) for private weight matrix multiplication.
Express LP certificate verification + DualMatrix as one Nova-foldable relation.
Formalize fixed-point soundness all six cases in Lean 4.
Benchmark end-to-end on MNIST and CIFAR-10.
Output: full paper targeting CCS, S&P, or USENIX Security.

### Target Benchmarks to Report
- Constraint count per image vs exact ReLU (aim: 100-200× reduction)
- Proving time for 1,000 / 10,000 images
- Proof size (should be constant regardless of test set size — Nova property)
- Verification time (target: < 1 second)
- Certified accuracy on MNIST at ε = 0.1, 0.3

### Known Engineering Challenge: Nova + Bulletproofs Unification

Nova folds a single relation repeatedly. Bulletproofs prove inner products
as a separate protocol. Using both requires expressing LP certificate
verification as ONE unified relation that Nova can fold, where the inner
product arguments are sub-steps inside that relation — not two separate
proof systems composed after the fact.

This is the main protocol design challenge of Phase 3-4. Options:
- Use Halo2 (IPA-based) which handles this more naturally than Nova
- Encode the Bulletproof inner product argument as a Nova-foldable circuit
- Use a single Pedersen-based IPA throughout (avoids the split entirely)

This is an open engineering problem — not a theoretical blocker, but
requires careful design before implementation begins.

### Critical Requirement: Find a Collaborator

This project requires expertise in two hard fields simultaneously:
- **Certification side** (DeepPoly, CROWN, abstract interpretation) — you have this
- **ZK engineering side** (Gnark, Nova, KZG, Bulletproofs) — you need this

Options:
- Contact FairProof authors (UC San Diego) — they solved the Gnark engineering already
- Contact ETH Zurich SRI — built DeepPoly, expressed interest in ZK combinations
- Find a CS PhD student specializing in applied cryptography at your institution

Do not underestimate this requirement. The ZK implementation is where most ML
researchers attempting this kind of work get stuck.

---

## Expected Results

| Metric | Expected |
| --- | --- |
| Proof size | < 10 KB (constant in N — Nova property) |
| Verification time | < 1 second |
| Proving time (MNIST, 10K images) | Hours (offline, one-time cost) |
| Certified accuracy at ε=0.1 | ~60-70% (matches DeepPoly baseline) |
| Constraints vs exact ReLU | ~100-200× fewer |

---

## Paper Pitch

> *"We present the first sound, compressed zero-knowledge system for private
> certified-accuracy auditing of neural networks. Our key insight is LP certificate
> verification: instead of re-running DeepPoly inside a ZK circuit, we verify
> pre-computed DeepPoly bounds using only linear arithmetic, reducing constraints by
> 100-200× versus exact-ReLU approaches. We prove a fixed-point soundness theorem
> showing that ZK-accepted certificates imply real robustness under finite-field
> arithmetic. Combined with Nova folding, our system produces a constant-size proof
> that a network achieves ≥ X% certified robustness at radius ε — without revealing
> model weights, test images, or labels."*

**Target venues:** CCS, S&P (IEEE), USENIX Security, ICLR (ML track)

---

## Key Papers to Read

### Foundation
- **DeepPoly** — Singh et al., POPL 2019.
  Abstract interpretation for NN certification. Our offline component.
  [PDF](https://ggndpsngh.github.io/files/DeepPoly.pdf)

- **α,β-CROWN** — Wang et al., NeurIPS 2021.
  State-of-the-art certified training. Understand bound propagation.
  [arXiv:2103.06624](https://arxiv.org/abs/2103.06624)

### Most Important Related Work
- **FairProof** — Yadav et al., ICML 2024.
  ZK proof of fairness certificate. Closest structural analog to our work.
  [arXiv:2402.12572](https://arxiv.org/abs/2402.12572)

- **Kaizen** — Abbaszadeh et al., CCS 2024.
  ZK proof of training. Proves correct training without revealing model/data.
  [ePrint:2024/162](https://eprint.iacr.org/2024/162)

### ZK Infrastructure
- **Nova** — Kothapalli et al., CRYPTO 2022.
  Folding scheme for incremental verifiable computation. Our aggregation layer.
  [arXiv:2107.04315](https://arxiv.org/abs/2107.04315)

- **TeleSparse** — PoPETS 2025.
  ZK-friendly sparsification for NN inference. Compare against for circuit size.
  [arXiv:2504.19274](https://arxiv.org/abs/2504.19274)

- **EZKL** — Open-source toolkit for ZK inference proofs.
  [github.com/zkonduit/ezkl](https://github.com/zkonduit/ezkl)

### Inspiration: Deployed ZK Systems
- **Zcash Protocol Specification** — Hopwood et al., 2025.
  The blueprint for production ZK systems. Study Sapling's circuit decomposition
  and Jubjub curve arithmetic — the direct model for our design.
  [protocol.pdf](https://zips.z.cash/protocol/protocol.pdf)

- **Bulletproofs** — Bünz et al., S&P 2018.
  As deployed in Monero for range proofs. Directly applicable to our final
  certification check. Production-proven, no trusted setup.
  [Monero implementation](https://github.com/monero-project/monero/blob/master/src/ringct/bulletproofs.cc)

### Survey
- **ZK Proof Based Verifiable ML Survey** — Peng et al., Feb 2025.
  Full map of the ZKML field 2017-2025.
  [arXiv:2502.18535](https://arxiv.org/abs/2502.18535)

### ZK Proof Systems
- **DualMatrix** — ePrint 2025/1768, Springer Cybersecurity 2026.
  Proves Y=W·X with both matrices private. O(N+n) prover, O(log n) proof,
  transparent setup. Used in Version C for private weight layer verification.
  [ePrint 2025/1768](https://eprint.iacr.org/2025/1768)

- **zkMatrix** — Cong et al., AsiaCCS 2024.
  Predecessor to DualMatrix. Useful for understanding the approach.
  [ePrint 2024/161](https://eprint.iacr.org/2024/161)

- **Halo2** — Zcash/ECC, 2021. Trustless ZK framework using IPA instead of KZG.
  No trusted setup. Used by zkSync, Polygon, Scroll. Our target framework.
  [github.com/zcash/halo2](https://github.com/zcash/halo2)

- **Lean 4 + Mathlib** — Proof assistant for formalizing the soundness theorem.
  Mathlib has interval arithmetic and floor/ceil lemmas we need directly.
  [leanprover-community.github.io](https://leanprover-community.github.io)

- **Gnark** — Go ZK framework. FairProof used this — reference implementation.
  [github.com/Consensys/gnark](https://github.com/consensys/gnark)

- **Bulletproofs** — Bünz et al., S&P 2018.
  Inner product arguments for linear arithmetic. Eliminates circuits for linear layers.
  [arXiv:1907.06381](https://arxiv.org/abs/1907.06381)

- **Nova** — Kothapalli et al., CRYPTO 2022.
  Folding scheme for IVC. Our aggregation layer over N images.
  [arXiv:2107.04315](https://arxiv.org/abs/2107.04315)

- **ZK AI Inference High Precision** — CCS 2025.
  Latest inference proof efficiency results to compare against.
  [ACM DL](https://dl.acm.org/doi/10.1145/3719027.3765056)

---

## Groups to Contact

| Group | Why |
|---|---|
| ETH Zurich SRI (Gagandeep Singh, Martin Vechev) | Built DeepPoly — ideal collaborators for the certification side |
| UC San Diego ML Group (FairProof authors) | Built the closest prior system — understand their circuit design |
| Consensys (Gnark team) | ZK engineering support |
