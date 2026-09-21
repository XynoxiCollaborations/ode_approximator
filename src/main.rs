use ordered_float::OrderedFloat;
use std::collections::BTreeMap;

type Time = OrderedFloat<f64>;
type Position = f64;
type Velocity = f64;
type Acceleration = f64;
type Spring = f64;
type Mass = f64;

#[derive(Debug, Clone)]
struct SpringSystem {
    spring_constant: Spring,
    values: BTreeMap<Time, MechanicalValues>,
    delta_time: Time,
    mass: Mass,
}

#[derive(Debug, Clone)]
struct MechanicalValues {
    position: Position,
    velocity: Velocity,
    acceleration: Acceleration,
}

impl MechanicalValues {
    fn new(position: Position, velocity: Velocity, acceleration: Acceleration) -> Self {
        Self {
            position,
            velocity,
            acceleration,
        }
    }
}

impl SpringSystem {
    fn new(spring_constant: Spring, position: Position, delta_time: Time, mass: Mass) -> Self {
        let mut values = BTreeMap::new();

        let a = -(spring_constant / mass) * position;
        let mech_val = MechanicalValues::new(position, 0.0, a);
        values.insert(OrderedFloat(0.0), mech_val);
        Self {
            values,
            spring_constant,
            mass,
            delta_time,
        }
    }
    fn a(&self, position: Position) -> Acceleration {
        let d = self.spring_constant;
        let m = self.mass;

        (-d / m) * position
    }

    fn last_a(&self) -> Acceleration {
        (*self
            .values
            .last_key_value()
            .expect("There should be a value")
            .1)
            .acceleration
    }

    fn last_v(&self) -> Velocity {
        (*self
            .values
            .last_key_value()
            .expect("There should be a value")
            .1)
            .velocity
    }

    fn last_x(&self) -> Position {
        (*self
            .values
            .last_key_value()
            .expect("There should be a value")
            .1)
            .position
    }

    fn last_t(&self) -> Time {
        *self
            .values
            .last_key_value()
            .expect("There should be a value")
            .0
    }
}

// Leapfrog
impl SpringSystem {
    fn lf_v(&self) -> Velocity {
        let v_i = self.last_v();
        let a_i = self.last_a();
        let a_i_next = self.a(self.lf_x());
        let dt = self.delta_time.0;

        v_i + 0.5 * (a_i + a_i_next) * dt
    }

    fn lf_x(&self) -> Position {
        let x_i = self.last_x();
        let v_i = self.last_v();
        let a_i = self.last_a();
        let dt = self.delta_time.0;

        x_i + v_i * dt + 0.5 * a_i * dt.powi(2)
    }

    fn lf_step(&mut self) {
        let x = self.lf_x();
        let v = self.lf_v();
        let a = self.a(x);
        let t = self.last_t() + self.delta_time.0;
        let mech_vals = MechanicalValues::new(x, v, a);
        self.values.insert(t, mech_vals);
    }
}

// Runge-Kutta 4th Order
impl SpringSystem {
    fn rk4_x(&self) -> Velocity {
        let v_i = self.last_v();

        let k_1 = v_i;
        let k_2 = v_i + 0.5 * self.delta_time.0 * k_1;
        let k_3 = v_i + 0.5 * self.delta_time.0 * k_2;
        let k_4 = v_i + self.delta_time.0 * k_3;

        v_i + (self.delta_time.0 / 6.0) * (k_1 + 2.0 * k_2 + 2.0 * k_3 + k_4)
    }

    fn rk4_v(&self) -> Velocity {
        let x_i = self.last_x();

        let k_1 = self.a(x_i);
        let k_2 = self.a(x_i + 0.5 * self.delta_time.0 * k_1);
        let k_3 = self.a(x_i + 0.5 * self.delta_time.0 * k_2);
        let k_4 = self.a(x_i + self.delta_time.0 * k_3);

        x_i + (self.delta_time.0 / 6.0) * (k_1 + 2.0 * k_2 + 2.0 * k_3 + k_4)
    }

    fn rk4_step(&mut self) {
        let x = self.rk4_x();
        let v = self.rk4_v();
        let a = self.a(x);
        let t = self.last_t() + self.delta_time.0;
        let mech_vals = MechanicalValues::new(x, v, a);
        self.values.insert(t, mech_vals);
    }
}

fn main() {
    let mut system = SpringSystem::new(-1.2, 1.0, OrderedFloat(1.0), 10.0);

    for _ in 0..100 {
        system.rk4_step();
    }

    dbg!(system.values);
}
