//! Fisher information metric on agent state space.
//!
//! The Fisher information matrix Gᵢⱼ = E[∂log p(x|θ)/∂θᵢ · ∂log p(x|θ)/∂θⱼ]
//! defines a Riemannian metric on the parameter space of agent distributions.
//!
//! Geodesics under this metric are the **optimal paths** for changing an agent's
//! state — they minimize the information cost of transitions.
//!
//! For our fleet: an agent's "state" is its budget allocation (γ, η).
//! The Fisher metric tells us the true information-theoretic distance between
//! two budget allocations, not just Euclidean distance.

use std::f64::consts::PI;

/// Gaussian distribution parameters (μ, σ) as agent state proxy.
#[derive(Debug, Clone)]
pub struct AgentState {
    pub id: usize,
    pub mu: Vec<f64>,     // mean (budget center)
    pub sigma: Vec<f64>,  // std dev (budget spread)
}

impl AgentState {
    pub fn new(id: usize, mu: Vec<f64>, sigma: Vec<f64>) -> Self {
        Self { id, mu, sigma }
    }
    pub fn dim(&self) -> usize { self.mu.len() }

    /// Fisher information matrix for diagonal Gaussian.
    /// G_μμ = 1/σ², G_σσ = 2/σ², G_μσ = 0
    pub fn fisher_matrix(&self) -> Vec<Vec<f64>> {
        let d = self.dim();
        let n = 2 * d; // mu_0..mu_{d-1}, sigma_0..sigma_{d-1}
        let mut g = vec![vec![0.0; n]; n];
        for i in 0..d {
            let s2 = self.sigma[i] * self.sigma[i];
            if s2 > 1e-12 {
                g[i][i] = 1.0 / s2;                 // G_μᵢμᵢ
                g[d + i][d + i] = 2.0 / s2;          // G_σᵢσᵢ
            }
        }
        g
    }

    /// KL divergence from self to other (diagonal Gaussian).
    pub fn kl_divergence(&self, other: &AgentState) -> f64 {
        let mut kl = 0.0;
        for i in 0..self.dim() {
            let s2_self = self.sigma[i].powi(2);
            let s2_other = other.sigma[i].powi(2);
            if s2_self > 1e-12 && s2_other > 1e-12 {
                kl += 0.5 * (s2_self / s2_other
                    + (other.mu[i] - self.mu[i]).powi(2) / s2_other
                    - 1.0
                    + (s2_other / s2_self).ln());
            }
        }
        kl
    }

    /// Fisher-Rao distance (closed form for 1D Gaussians).
    /// d(p,q) = √2 · arctan(|μ_p - μ_q| / √(σ_p² + σ_q²))... 
    /// Simplified for diagonal: sum of 1D distances.
    pub fn fisher_rao_distance(&self, other: &AgentState) -> f64 {
        let mut d2 = 0.0;
        for i in 0..self.dim() {
            d2 += fisher_rao_1d(
                self.mu[i], self.sigma[i],
                other.mu[i], other.sigma[i],
            ).powi(2);
        }
        d2.sqrt()
    }

    /// Pack parameters into a flat vector [μ₀, μ₁, ..., σ₀, σ₁, ...].
    pub fn to_params(&self) -> Vec<f64> {
        let mut p = self.mu.clone();
        p.extend(self.sigma.iter().map(|s| s.max(0.01))); // clamp sigma
        p
    }

    /// Unpack parameters from flat vector.
    pub fn from_params(id: usize, params: &[f64]) -> Self {
        let d = params.len() / 2;
        let mu = params[0..d].to_vec();
        let sigma = params[d..2*d].to_vec();
        Self { id, mu, sigma }
    }
}

/// Fisher-Rao distance between two 1D Gaussians.
pub fn fisher_rao_1d(mu1: f64, sigma1: f64, mu2: f64, sigma2: f64) -> f64 {
    let s1 = sigma1.max(0.01);
    let s2 = sigma2.max(0.01);
    let ratio = s1.max(s2) / s1.min(s2);
    let term1 = 2.0 * ratio.ln();
    let term2 = 4.0 * (mu1 - mu2).powi(2) / (s1.powi(2) + s2.powi(2));
    (term1 + term2).sqrt().min(PI) // capped at π for stability
}

/// Geodesic between two agent states (linear interpolation in Fisher metric).
pub fn geodesic(a: &AgentState, b: &AgentState, n_points: usize) -> Vec<AgentState> {
    let pa = a.to_params();
    let pb = b.to_params();
    let mut path = vec![];
    for k in 0..=n_points {
        let t = k as f64 / n_points as f64;
        let params: Vec<f64> = pa.iter().zip(pb.iter())
            .map(|(a_i, b_i)| a_i * (1.0 - t) + b_i * t)
            .collect();
        path.push(AgentState::from_params(k, &params));
    }
    path
}

/// Geodesic length (sum of Fisher-Rao distances along path).
pub fn geodesic_length(path: &[AgentState]) -> f64 {
    let mut total = 0.0;
    for i in 1..path.len() {
        total += path[i - 1].fisher_rao_distance(&path[i]);
    }
    total
}

/// Fleet state — collection of agent states.
#[derive(Debug, Clone)]
pub struct FleetState {
    pub agents: Vec<AgentState>,
}

impl FleetState {
    pub fn new(agents: Vec<AgentState>) -> Self { Self { agents } }

    /// Average pairwise Fisher-Rao distance.
    pub fn average_pairwise_distance(&self) -> f64 {
        let n = self.agents.len();
        if n < 2 { return 0.0; }
        let mut total = 0.0; let mut count = 0;
        for i in 0..n { for j in (i+1)..n {
            total += self.agents[i].fisher_rao_distance(&self.agents[j]);
            count += 1;
        }}
        total / count as f64
    }

    /// Maximum pairwise distance (diameter).
    pub fn diameter(&self) -> f64 {
        let mut max_d = 0.0_f64;
        for i in 0..self.agents.len() { for j in (i+1)..self.agents.len() {
            max_d = max_d.max(self.agents[i].fisher_rao_distance(&self.agents[j]));
        }}
        max_d
    }

    /// Fleet centroid (mean of parameters).
    pub fn centroid(&self) -> AgentState {
        let d = self.agents[0].dim();
        let n = self.agents.len() as f64;
        let mut mu = vec![0.0; d];
        let mut sigma = vec![0.0; d];
        for a in &self.agents {
            for i in 0..d {
                mu[i] += a.mu[i] / n;
                sigma[i] += a.sigma[i] / n;
            }
        }
        AgentState::new(0, mu, sigma)
    }

    /// Information radius: average distance from centroid.
    pub fn information_radius(&self) -> f64 {
        let c = self.centroid();
        self.agents.iter().map(|a| a.fisher_rao_distance(&c)).sum::<f64>() / self.agents.len() as f64
    }

    /// Fisher information volume (det of average Fisher matrix, log scale).
    pub fn information_volume(&self) -> f64 {
        let avg = self.centroid();
        let g = avg.fisher_matrix();
        let n = g.len();
        let mut log_det = 0.0;
        for i in 0..n {
            if g[i][i] > 1e-12 { log_det += g[i][i].ln(); }
        }
        log_det
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn agent_1d(id: usize, mu: f64, sigma: f64) -> AgentState {
        AgentState::new(id, vec![mu], vec![sigma])
    }

    #[test]
    fn test_fisher_matrix_diagonal() {
        let a = agent_1d(0, 0.0, 1.0);
        let g = a.fisher_matrix();
        assert!((g[0][0] - 1.0).abs() < 1e-10, "G_μμ = 1/σ² = 1");
        assert!((g[1][1] - 2.0).abs() < 1e-10, "G_σσ = 2/σ² = 2");
        assert!(g[0][1].abs() < 1e-10, "G_μσ = 0");
    }

    #[test]
    fn test_fisher_matrix_scaling() {
        let a = agent_1d(0, 0.0, 2.0);
        let g = a.fisher_matrix();
        assert!((g[0][0] - 0.25).abs() < 1e-10, "G_μμ = 1/4");
        assert!((g[1][1] - 0.5).abs() < 1e-10, "G_σσ = 2/4 = 0.5");
    }

    #[test]
    fn test_kl_self_zero() {
        let a = agent_1d(0, 1.0, 2.0);
        assert!(a.kl_divergence(&a) < 1e-10, "KL(p||p) = 0");
    }

    #[test]
    fn test_kl_asymmetric() {
        let a = agent_1d(0, 0.0, 1.0);
        let b = agent_1d(1, 2.0, 3.0);
        let kl_ab = a.kl_divergence(&b);
        let kl_ba = b.kl_divergence(&a);
        assert!((kl_ab - kl_ba).abs() > 0.01, "KL should be asymmetric");
    }

    #[test]
    fn test_kl_positive() {
        let a = agent_1d(0, 0.0, 1.0);
        let b = agent_1d(1, 5.0, 1.0);
        assert!(a.kl_divergence(&b) > 0.0);
    }

    #[test]
    fn test_fisher_rao_same() {
        let a = agent_1d(0, 1.0, 2.0);
        assert!(a.fisher_rao_distance(&a) < 1e-10);
    }

    #[test]
    fn test_fisher_rao_different() {
        let a = agent_1d(0, 0.0, 1.0);
        let b = agent_1d(1, 5.0, 2.0);
        assert!(a.fisher_rao_distance(&b) > 0.0);
    }

    #[test]
    fn test_fisher_rao_triangle_inequality() {
        let a = agent_1d(0, 0.0, 1.0);
        let b = agent_1d(1, 3.0, 1.5);
        let c = agent_1d(2, 6.0, 2.0);
        let ab = a.fisher_rao_distance(&b);
        let bc = b.fisher_rao_distance(&c);
        let ac = a.fisher_rao_distance(&c);
        assert!(ac <= ab + bc + 0.01, "Triangle: {} ≤ {} + {}", ac, ab, bc);
    }

    #[test]
    fn test_geodesic_endpoints() {
        let a = agent_1d(0, 0.0, 1.0);
        let b = agent_1d(1, 5.0, 2.0);
        let path = geodesic(&a, &b, 10);
        assert_eq!(path.len(), 11);
        assert!((path[0].mu[0] - 0.0).abs() < 1e-10);
        assert!((path[10].mu[0] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_geodesic_length_positive() {
        let a = agent_1d(0, 0.0, 1.0);
        let b = agent_1d(1, 5.0, 2.0);
        let path = geodesic(&a, &b, 20);
        let len = geodesic_length(&path);
        assert!(len > 0.0);
    }

    #[test]
    fn test_fleet_average_distance() {
        let fleet = FleetState::new(vec![
            agent_1d(0, 0.0, 1.0),
            agent_1d(1, 5.0, 1.0),
            agent_1d(2, 10.0, 1.0),
        ]);
        let avg = fleet.average_pairwise_distance();
        assert!(avg > 0.0);
    }

    #[test]
    fn test_fleet_diameter() {
        let fleet = FleetState::new(vec![
            agent_1d(0, 0.0, 1.0),
            agent_1d(1, 5.0, 1.0),
            agent_1d(2, 10.0, 1.0),
        ]);
        let d = fleet.diameter();
        assert!(d > 0.0);
        // d(0,10) should be largest
        let d01 = fleet.agents[0].fisher_rao_distance(&fleet.agents[1]);
        assert!(d >= d01 - 0.01);
    }

    #[test]
    fn test_centroid() {
        let fleet = FleetState::new(vec![
            agent_1d(0, 0.0, 1.0),
            agent_1d(1, 10.0, 3.0),
        ]);
        let c = fleet.centroid();
        assert!((c.mu[0] - 5.0).abs() < 1e-10);
        assert!((c.sigma[0] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_information_radius() {
        let tight = FleetState::new(vec![
            agent_1d(0, 1.0, 1.0),
            agent_1d(1, 1.1, 1.0),
        ]);
        let spread = FleetState::new(vec![
            agent_1d(0, 0.0, 1.0),
            agent_1d(1, 10.0, 1.0),
        ]);
        assert!(tight.information_radius() < spread.information_radius());
    }

    #[test]
    fn test_information_volume() {
        let narrow = FleetState::new(vec![agent_1d(0, 0.0, 0.5)]);
        let wide = FleetState::new(vec![agent_1d(0, 0.0, 2.0)]);
        // Narrow sigma → larger Fisher info → larger log det
        assert!(narrow.information_volume() > wide.information_volume());
    }

    #[test]
    fn test_multidim() {
        let a = AgentState::new(0, vec![0.0, 0.0], vec![1.0, 1.0]);
        let b = AgentState::new(1, vec![3.0, 4.0], vec![1.0, 1.0]);
        let d = a.fisher_rao_distance(&b);
        assert!(d > 0.0);
    }

    #[test]
    fn test_params_roundtrip() {
        let a = AgentState::new(0, vec![1.0, 2.0], vec![0.5, 1.5]);
        let p = a.to_params();
        let b = AgentState::from_params(0, &p);
        assert!((a.mu[0] - b.mu[0]).abs() < 1e-10);
        assert!((a.sigma[1] - b.sigma[1]).abs() < 1e-10);
    }
}
