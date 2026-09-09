use ordered_float::OrderedFloat;
use std::collections::BTreeMap;
use std::ops::{Add, Mul};

type TimeFloat = OrderedFloat<f64>;
type ValueFloat = f64;

/// The Entire system, with saved datapoints
#[derive(Debug)]
struct System<F, const N: usize>
where
    F: Fn(TimeFloat, ValuesVector<N>) -> ValuesVector<N>,
{
    /// The calculated datapoints. Is a BTreeMap for ordering
    datapoints: BTreeMap<TimeFloat, ValuesVector<N>>,
    /// The differntial equation which gives new values for next iteration.
    func: F,
    /// How Many Seconds per step?
    h: TimeFloat,
}

impl<F, const N: usize> System<F, N>
where
    F: Fn(TimeFloat, ValuesVector<N>) -> ValuesVector<N>,
{
    /// Create a new system with the given initial values y_0, h, and a function.
    fn new(initial_values: ValuesVector<N>, func: F, h: TimeFloat) -> Self {
        let mut datapoints = BTreeMap::new();
        datapoints.insert(OrderedFloat(0.0), initial_values);
        Self {
            datapoints,
            func,
            h,
        }
    }

    /// Perform one H step for Runge Kutta Order 4 method.
    fn rk4_step(&mut self) {
        let (&t_n, &y_n) = self
            .datapoints
            .last_key_value()
            .expect("There should be at least one value");
        let k_1 = (self.func)(t_n, y_n);
        let k_2 = (self.func)(t_n + (self.h / 2.0), y_n + k_1 * (self.h / 2.0));
        let k_3 = (self.func)(t_n + (self.h / 2.0), y_n + k_2 * (self.h / 2.0));
        let k_4 = (self.func)(t_n + self.h, y_n + k_3 * self.h);

        let t_n_new = t_n + self.h;
        let y_n_new = y_n + (k_1 + k_2 * 2.0 + k_3 * 2.0 + k_4) * (self.h / 6.0);

        self.datapoints.insert(t_n_new, y_n_new);
    }

    /// Perform N steps of Runge Kutta Order 4 method
    fn rk4_n_times(&mut self, n: usize) {
        for _ in 0..n {
            self.rk4_step();
        }
    }

    /// Perform Runge Kutta Order 4 method until `target_time` is reached.
    fn rk4_until(&mut self, target_time: TimeFloat) {
        let (&mut t, _) = self
            .datapoints
            .last_key_value()
            .expect("There should be at least one value");

        while t < target_time {
            t += self.h;
            self.rk4_step();
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct ValuesVector<const N: usize>([ValueFloat; N]);

impl<const N: usize> Add for ValuesVector<N> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self(std::array::from_fn(|i| self.0[i] + other.0[i]))
    }
}

impl<const N: usize> Mul<f64> for ValuesVector<N> {
    type Output = Self;

    fn mul(self, factor: f64) -> Self {
        Self(self.0.map(|v| v * factor))
    }
}

impl<const N: usize> Mul<OrderedFloat<f64>> for ValuesVector<N> {
    type Output = Self;

    fn mul(self, factor: OrderedFloat<f64>) -> Self {
        Self(self.0.map(|v| v * factor.0))
    }
}

fn main() {
    todo!("Create a simulation of pendula");
}
