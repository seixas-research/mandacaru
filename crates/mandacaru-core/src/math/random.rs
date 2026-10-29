//! Deterministic local random streams; no process-global RNG state.
pub struct Rng(pub u64);
impl Rng {
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut value = self.0;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d049bb133111eb);
        value ^ (value >> 31)
    }
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    pub fn normal(&mut self) -> f64 {
        let u = loop {
            let value = self.uniform();
            if value > 0. {
                break value;
            }
        };
        (-2. * u.ln()).sqrt() * (2. * std::f64::consts::PI * self.uniform()).cos()
    }
    pub fn index(&mut self, upper: usize) -> usize {
        assert!(upper > 0);
        let upper = upper as u64;
        let zone = u64::MAX - u64::MAX % upper;
        loop {
            let value = self.next_u64();
            if value < zone {
                return (value % upper) as usize;
            }
        }
    }
}
