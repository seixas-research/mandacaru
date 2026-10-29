//! Matrix reverse derivatives for small learned scientific models.
//! Every retained value and adjoint belongs to the calculation budget.
use crate::{finite, product, Context, Error, Reservation, Result};
use nalgebra::DMatrix;
pub type Id = usize;
type Matrix = DMatrix<f64>;
#[derive(Clone)]
enum Op {
    Leaf,
    Constant,
    Matmul(Id, Id),
    Add(Id, Id),
    Tanh(Id),
    Concat(Vec<Id>),
    Repeat(Id),
    Gather(Id, Vec<usize>),
    Scale(Id, Vec<f64>),
    Normalize(Id, f64),
}
struct Node {
    value: Matrix,
    gradient: Matrix,
    operation: Op,
    requires_gradient: bool,
    _memory: Reservation,
}
pub struct Tape {
    context: Context,
    nodes: Vec<Node>,
}
fn invalid(message: &str) -> Error {
    Error::Invalid(message.into())
}
impl Tape {
    pub fn new(context: &Context) -> Self {
        Self {
            context: context.clone(),
            nodes: vec![],
        }
    }
    pub fn value(&self, id: Id) -> &Matrix {
        &self.nodes[id].value
    }
    pub fn gradient(&self, id: Id) -> &Matrix {
        &self.nodes[id].gradient
    }
    fn memory(&self, rows: usize, columns: usize, metadata: usize) -> Result<Reservation> {
        self.context.reserve(
            product(&[rows, columns, 16])?
                .checked_add(metadata)
                .and_then(|v| v.checked_add(256))
                .ok_or(Error::Overflow)?,
        )
    }
    fn push(&mut self, value: Matrix, operation: Op, memory: Reservation) -> Id {
        let requires_gradient = match &operation {
            Op::Leaf => true,
            Op::Constant => false,
            Op::Matmul(a, b) | Op::Add(a, b) => {
                self.nodes[*a].requires_gradient || self.nodes[*b].requires_gradient
            }
            Op::Concat(parts) => parts.iter().any(|&id| self.nodes[id].requires_gradient),
            Op::Tanh(a)
            | Op::Repeat(a)
            | Op::Gather(a, _)
            | Op::Scale(a, _)
            | Op::Normalize(a, _) => self.nodes[*a].requires_gradient,
        };
        let gradient = Matrix::zeros(value.nrows(), value.ncols());
        let id = self.nodes.len();
        self.nodes.push(Node {
            value,
            gradient,
            operation,
            requires_gradient,
            _memory: memory,
        });
        id
    }
    pub fn leaf(&mut self, value: &Matrix) -> Result<Id> {
        finite(value.as_slice(), "model matrix")?;
        let memory = self.memory(value.nrows(), value.ncols(), 0)?;
        Ok(self.push(value.clone(), Op::Leaf, memory))
    }
    pub fn constant(&mut self, value: &Matrix) -> Result<Id> {
        finite(value.as_slice(), "constant model matrix")?;
        let memory = self.memory(value.nrows(), value.ncols(), 0)?;
        Ok(self.push(value.clone(), Op::Constant, memory))
    }
    pub fn matmul(&mut self, a: Id, b: Id) -> Result<Id> {
        if self.value(a).ncols() != self.value(b).nrows() {
            return Err(invalid("matrix derivative product dimensions disagree"));
        }
        let memory = self.memory(self.value(a).nrows(), self.value(b).ncols(), 0)?;
        let value = self.value(a) * self.value(b);
        Ok(self.push(value, Op::Matmul(a, b), memory))
    }
    pub fn add(&mut self, a: Id, b: Id) -> Result<Id> {
        let x = self.value(a);
        let y = self.value(b);
        if x.ncols() != y.ncols() || (x.nrows() != y.nrows() && y.nrows() != 1) {
            return Err(invalid("matrix derivative sum dimensions disagree"));
        }
        let memory = self.memory(x.nrows(), x.ncols(), 0)?;
        let value = Matrix::from_fn(x.nrows(), x.ncols(), |r, c| {
            x[(r, c)] + y[(if y.nrows() == 1 { 0 } else { r }, c)]
        });
        Ok(self.push(value, Op::Add(a, b), memory))
    }
    pub fn tanh(&mut self, a: Id) -> Result<Id> {
        let memory = self.memory(self.value(a).nrows(), self.value(a).ncols(), 0)?;
        let value = self.value(a).map(f64::tanh);
        Ok(self.push(value, Op::Tanh(a), memory))
    }
    pub fn concat(&mut self, parts: &[Id]) -> Result<Id> {
        let rows = parts
            .first()
            .map(|&id| self.value(id).nrows())
            .ok_or_else(|| invalid("concatenation needs a matrix"))?;
        if parts.iter().any(|&id| self.value(id).nrows() != rows) {
            return Err(invalid("concatenated rows must match"));
        }
        let columns = parts.iter().try_fold(0usize, |n, &id| {
            n.checked_add(self.value(id).ncols()).ok_or(Error::Overflow)
        })?;
        let memory = self.memory(rows, columns, product(&[parts.len(), 8])?)?;
        let mut value = Matrix::zeros(rows, columns);
        let mut first = 0;
        for &id in parts {
            let width = self.value(id).ncols();
            value.columns_mut(first, width).copy_from(self.value(id));
            first += width;
        }
        Ok(self.push(value, Op::Concat(parts.to_vec()), memory))
    }
    pub fn repeat(&mut self, a: Id, rows: usize) -> Result<Id> {
        let x = self.value(a);
        if x.nrows() != 1 {
            return Err(invalid("broadcasting requires one row"));
        }
        let memory = self.memory(rows, x.ncols(), 0)?;
        let value = Matrix::from_fn(rows, x.ncols(), |_, c| x[(0, c)]);
        Ok(self.push(value, Op::Repeat(a), memory))
    }
    pub fn gather(&mut self, a: Id, indices: &[usize]) -> Result<Id> {
        let x = self.value(a);
        if indices.iter().any(|&i| i >= x.nrows()) {
            return Err(invalid("gather row exceeds model matrix"));
        }
        let memory = self.memory(indices.len(), x.ncols(), product(&[indices.len(), 8])?)?;
        let value = Matrix::from_fn(indices.len(), x.ncols(), |r, c| x[(indices[r], c)]);
        Ok(self.push(value, Op::Gather(a, indices.to_vec()), memory))
    }
    pub fn scale_rows(&mut self, a: Id, weights: &[f64]) -> Result<Id> {
        let x = self.value(a);
        if weights.len() != x.nrows() {
            return Err(invalid("row scales must match the model matrix"));
        }
        finite(weights, "row scales")?;
        let memory = self.memory(x.nrows(), x.ncols(), product(&[weights.len(), 8])?)?;
        let value = Matrix::from_fn(x.nrows(), x.ncols(), |r, c| x[(r, c)] * weights[r]);
        Ok(self.push(value, Op::Scale(a, weights.to_vec()), memory))
    }
    pub fn normalize_rows(&mut self, a: Id, epsilon: f64) -> Result<Id> {
        let x = self.value(a);
        if x.ncols() == 0 || !epsilon.is_finite() || epsilon <= 0. {
            return Err(invalid(
                "normalization requires columns and positive epsilon",
            ));
        }
        let memory = self.memory(x.nrows(), x.ncols(), 0)?;
        let mut value = x.clone();
        for r in 0..x.nrows() {
            let mean = x.row(r).iter().sum::<f64>() / x.ncols() as f64;
            let sigma = (x.row(r).iter().map(|v| (v - mean).powi(2)).sum::<f64>()
                / x.ncols() as f64)
                .sqrt();
            for c in 0..x.ncols() {
                value[(r, c)] = (x[(r, c)] - mean) / (sigma + epsilon);
            }
        }
        Ok(self.push(value, Op::Normalize(a, epsilon), memory))
    }
    pub fn backward(&mut self, output: Id, seed: &Matrix) -> Result<()> {
        self.backward_many(&[(output, seed.clone())])
    }
    pub fn backward_many(&mut self, outputs: &[(Id, Matrix)]) -> Result<()> {
        if outputs.is_empty() {
            return Err(invalid("reverse derivatives need an output"));
        }
        for (output, seed) in outputs {
            if *output >= self.nodes.len() || seed.shape() != self.value(*output).shape() {
                return Err(invalid("loss adjoint must match the model output"));
            }
            finite(seed.as_slice(), "loss adjoint")?;
        }
        let cells = self.nodes.iter().map(|n| n.value.len()).max().unwrap_or(0);
        let _workspace = self.context.reserve(
            product(&[cells, 64])?
                .checked_add(65536)
                .ok_or(Error::Overflow)?,
        )?;
        for node in &mut self.nodes {
            node.gradient.fill(0.);
        }
        for (output, seed) in outputs {
            self.nodes[*output].gradient += seed;
        }
        let output = outputs.iter().map(|(id, _)| *id).max().unwrap();
        for id in (0..=output).rev() {
            if !self.nodes[id].requires_gradient {
                continue;
            }
            let operation = self.nodes[id].operation.clone();
            let gradient = self.nodes[id].gradient.clone();
            match operation {
                Op::Leaf | Op::Constant => {}
                Op::Matmul(a, b) => {
                    if self.nodes[a].requires_gradient {
                        let da = &gradient * self.value(b).transpose();
                        self.nodes[a].gradient += da;
                    }
                    if self.nodes[b].requires_gradient {
                        let db = self.value(a).transpose() * &gradient;
                        self.nodes[b].gradient += db;
                    }
                }
                Op::Add(a, b) => {
                    self.nodes[a].gradient += &gradient;
                    if self.value(b).nrows() == gradient.nrows() {
                        self.nodes[b].gradient += &gradient;
                    } else {
                        for c in 0..gradient.ncols() {
                            self.nodes[b].gradient[(0, c)] +=
                                gradient.column(c).iter().sum::<f64>();
                        }
                    }
                }
                Op::Tanh(a) => {
                    for (j, &g) in gradient.iter().enumerate() {
                        self.nodes[a].gradient[j] += g * (1. - self.nodes[id].value[j].powi(2));
                    }
                }
                Op::Concat(parts) => {
                    let mut first = 0;
                    for a in parts {
                        let width = self.value(a).ncols();
                        if self.nodes[a].requires_gradient {
                            self.nodes[a].gradient += gradient.columns(first, width);
                        }
                        first += width;
                    }
                }
                Op::Repeat(a) => {
                    for c in 0..gradient.ncols() {
                        self.nodes[a].gradient[(0, c)] += gradient.column(c).iter().sum::<f64>();
                    }
                }
                Op::Gather(a, indices) => {
                    for (r, &target) in indices.iter().enumerate() {
                        for c in 0..gradient.ncols() {
                            self.nodes[a].gradient[(target, c)] += gradient[(r, c)];
                        }
                    }
                }
                Op::Scale(a, weights) => {
                    for r in 0..gradient.nrows() {
                        for c in 0..gradient.ncols() {
                            self.nodes[a].gradient[(r, c)] += gradient[(r, c)] * weights[r];
                        }
                    }
                }
                Op::Normalize(a, epsilon) => {
                    let x = self.value(a);
                    let width = x.ncols() as f64;
                    let mut derivative = Matrix::zeros(x.nrows(), x.ncols());
                    for r in 0..x.nrows() {
                        let mean = x.row(r).iter().sum::<f64>() / width;
                        let sigma = (x.row(r).iter().map(|v| (v - mean).powi(2)).sum::<f64>()
                            / width)
                            .sqrt();
                        let mean_g = gradient.row(r).iter().sum::<f64>() / width;
                        let cross = (0..x.ncols())
                            .map(|c| gradient[(r, c)] * (x[(r, c)] - mean))
                            .sum::<f64>()
                            / width;
                        for c in 0..x.ncols() {
                            derivative[(r, c)] = (gradient[(r, c)] - mean_g) / (sigma + epsilon)
                                - if sigma > 0. {
                                    (x[(r, c)] - mean) * cross / (sigma * (sigma + epsilon).powi(2))
                                } else {
                                    0.
                                };
                        }
                    }
                    self.nodes[a].gradient += derivative;
                }
            }
        }
        if self
            .nodes
            .iter()
            .flat_map(|n| n.gradient.iter())
            .any(|g| !g.is_finite())
        {
            return Err(Error::Kernel("nonfinite model derivative".into()));
        }
        Ok(())
    }
}
