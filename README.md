# si-information-geodesic

> **Proof of Concept:** Fisher information metric on agent state space — geodesics trace the optimal (minimum-information-cost) path between budget allocations.

## The Insight

Euclidean distance between two agent states (budget allocations) is misleading. A change from σ=1 to σ=1.1 is "farther" in information space than from σ=10 to σ=10.1.

The **Fisher information matrix** G defines a Riemannian metric on the space of agent distributions:

| Parameter | Fisher Info | Meaning |
|-----------|------------|---------|
| G_μμ = 1/σ² | Mean precision | Narrow agents are harder to shift |
| G_σσ = 2/σ² | Spread precision | Narrow agents resist spread change |
| G_μσ = 0 | Independence | Mean and spread are orthogonal |

**Geodesics** under this metric are the optimal paths — they minimize the total Fisher information cost of transitioning between states.

## What This Proves

1. **Fisher-Rao distance > Euclidean** for non-uniform agents
2. **KL divergence is asymmetric** — shifting a narrow agent is more expensive
3. **Triangle inequality holds** — Fisher-Rao is a true metric
4. **Information radius** quantifies fleet spread in information space
5. **Information volume** (det of Fisher matrix) measures fleet "information capacity"

## Usage

```rust
use si_information_geodesic::*;

// Agent as 1D Gaussian (budget center μ, spread σ)
let a = AgentState::new(0, vec![0.0], vec![1.0]);
let b = AgentState::new(1, vec![5.0], vec![2.0]);

// Fisher information matrix
let g = a.fisher_matrix();
// G_μμ = 1, G_σσ = 2, G_μσ = 0

// Distances
let kl = a.kl_divergence(&b);     // asymmetric
let fr = a.fisher_rao_distance(&b); // symmetric metric

// Optimal path (geodesic)
let path = geodesic(&a, &b, 20);
let cost = geodesic_length(&path);

// Fleet analysis
let fleet = FleetState::new(vec![a, b]);
println!("Avg pairwise: {}", fleet.average_pairwise_distance());
println!("Diameter: {}", fleet.diameter());
println!("Info radius: {}", fleet.information_radius());
println!("Info volume: {}", fleet.information_volume());
```

## Modules

- `AgentState` — agent as diagonal Gaussian (μ = budget center, σ = budget spread)
- `fisher_matrix()` — Gᵢⱼ for diagonal Gaussian
- `kl_divergence()` — KL(p||q) between agent distributions
- `fisher_rao_distance()` — true metric distance (symmetric)
- `geodesic()` — path between two states in Fisher metric
- `geodesic_length()` — total information cost of path
- `FleetState` — collection of agents with pairwise analysis
- `information_radius()` — average distance from centroid
- `information_volume()` — log det of Fisher matrix (information capacity)

## Connection to Conservation Law

Budget transitions must respect γ + η = C. In Fisher geometry:
- The constraint manifold is a **submanifold** of the full state space
- Geodesics on the constraint manifold are the optimal budget-respecting paths
- The Fisher metric tells you the *true cost* of budget reallocation
- Moving budget from a narrow (precise) agent is more expensive than from a wide (uncertain) one

## Mathematical Background

### Fisher Information Matrix
For diagonal Gaussian N(μ, σ²):
G = diag(1/σ², 2/σ²) per dimension

### Fisher-Rao Distance (1D)
d(p,q) = √(2·ln(σ_q/σ_p + σ_p/σ_q) + 4·(μ_p-μ_q)²/(σ_p²+σ_q²))

For our implementation: capped at π for stability.

### KL Divergence (Diagonal Gaussian)
KL(p||q) = Σᵢ [½(σ²_pᵢ/σ²_qᵢ + (μ_qᵢ-μ_pᵢ)²/σ²_qᵢ - 1 + ln(σ²_qᵢ/σ²_pᵢ))]

## Tests: 17

Covers: Fisher matrix values, scaling, KL self-zero, asymmetry, Fisher-Rao same/different, triangle inequality, geodesic endpoints/length, fleet distances, diameter, centroid, information radius ordering, information volume ordering, multidimensional, parameter roundtrip.

## License

MIT
