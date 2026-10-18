use anyhow::Result;
use arrow_array::{ArrayRef, Float64Array, RecordBatch};
use ordered_float::OrderedFloat;
use parquet::arrow::ArrowWriter;
use std::collections::BTreeMap;
use std::{fs::File, sync::Arc};

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

#[derive(Debug, Clone, Copy)]
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

    fn write_to_file(&self, path: &str) -> Result<()> {
        let times = self.values.keys().map(|&t| t.0).collect::<Float64Array>();

        let positions = self
            .values
            .values()
            .map(|&mv| mv.position)
            .collect::<Float64Array>();

        let velocities = self
            .values
            .values()
            .map(|&mv| mv.velocity)
            .collect::<Float64Array>();

        let accelerations = self
            .values
            .values()
            .map(|&mv| mv.acceleration)
            .collect::<Float64Array>();

        let batch = RecordBatch::try_from_iter([
            ("time", Arc::new(times) as ArrayRef),
            ("position", Arc::new(positions) as ArrayRef),
            ("velocity", Arc::new(velocities) as ArrayRef),
            ("acceleration", Arc::new(accelerations) as ArrayRef),
        ])?;

        let file = File::create(path)?;

        let mut writer = ArrowWriter::try_new(file, batch.schema(), None)?;
        writer.write(&batch)?;
        writer.close()?;

        Ok(())
    }
}

// Leapfrog
impl SpringSystem {
    fn lf_new(spring_constant: Spring, position: Position, delta_time: Time, mass: Mass) -> Self {
        let mut values = BTreeMap::new();

        let a = -(spring_constant / mass) * position;
        let v = -0.5 * a * delta_time.0;
        let mech_val = MechanicalValues::new(position, v, a);
        values.insert(OrderedFloat(0.0), mech_val);

        Self {
            values,
            spring_constant,
            mass,
            delta_time,
        }
    }

    fn lf_v(&self) -> Velocity {
        let v_i = self.last_v();
        let a_i = self.last_a();
        let dt = self.delta_time.0;

        v_i + a_i * dt
    }

    fn lf_x(&self) -> Position {
        let x_i = self.last_x();
        let v_i = self.lf_v();
        let dt = self.delta_time.0;

        x_i + v_i * dt
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
    fn rk4_step(&mut self) {
        let x_i = self.last_x();
        let v_i = self.last_v();
        let dt = self.delta_time.0;

        let k1_x = v_i;
        let k1_v = self.a(x_i);

        let k2_x = v_i + 0.5 * dt * k1_v;
        let k2_v = self.a(x_i + 0.5 * dt * k1_x);

        let k3_x = v_i + 0.5 * dt * k2_v;
        let k3_v = self.a(x_i + 0.5 * dt * k2_x);

        let k4_x = v_i + dt * k3_v;
        let k4_v = self.a(x_i + dt * k3_x);

        let x = x_i + (dt / 6.0) * (k1_x + 2.0 * k2_x + 2.0 * k3_x + k4_x);
        let v = v_i + (dt / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v);
        let a = self.a(x);

        let t = self.last_t() + dt;
        let mech_vals = MechanicalValues::new(x, v, a);
        self.values.insert(t, mech_vals);
    }
}

// Euler Method
impl SpringSystem {
    fn euler_x(&self) -> Position {
        let x_i = self.last_x();
        let v_i = self.last_v();

        x_i + v_i * self.delta_time.0
    }

    fn euler_v(&self) -> Velocity {
        let v_i = self.last_v();
        let a_i = self.last_a();

        v_i + a_i * self.delta_time.0
    }

    fn euler_step(&mut self) {
        let x = self.euler_x();
        let v = self.euler_v();
        let a = self.a(x);
        let t = self.last_t() + self.delta_time.0;
        let mech_vals = MechanicalValues::new(x, v, a);
        self.values.insert(t, mech_vals);
    }
}

// Euler-Cromer method
impl SpringSystem {
    fn ec_x(&self) -> Position {
        let x_i = self.last_x();
        let v_next = self.ec_v();

        x_i + v_next * self.delta_time.0
    }

    fn ec_v(&self) -> Velocity {
        let v_i = self.last_v();
        let a_i = self.last_a();

        v_i + a_i * self.delta_time.0
    }

    fn ec_step(&mut self) {
        let x = self.ec_x();
        let v = self.ec_v();
        let a = self.a(x);
        let t = self.last_t() + self.delta_time.0;
        let mech_vals = MechanicalValues::new(x, v, a);
        self.values.insert(t, mech_vals);
    }
}

// Heun's method
impl SpringSystem {
    fn heun_x(&self) -> Position {
        let x_i = self.last_x();
        let v_i = self.last_v();
        let a_i = self.last_a();

        let v_predict = v_i + a_i * self.delta_time.0;

        x_i + self.delta_time.0 / 2.0 * (v_i + v_predict)
    }

    fn heun_v(&self) -> Velocity {
        let x_i = self.last_x();
        let v_i = self.last_v();
        let a_i = self.last_a();

        let x_predict = x_i + v_i * self.delta_time.0;
        let a_predict = self.a(x_predict);

        v_i + self.delta_time.0 / 2.0 * (a_i + a_predict)
    }

    fn heun_step(&mut self) {
        let x = self.heun_x();
        let v = self.heun_v();
        let a = self.a(x);
        let t = self.last_t() + self.delta_time.0;
        let mech_vals = MechanicalValues::new(x, v, a);
        self.values.insert(t, mech_vals);
    }
}

fn main() -> Result<()> {
    let steps = 40;

    let spring_constant = 1.0;
    let position = 1.0;
    let time = OrderedFloat(0.5);
    let mass = 1.0;

    // Leapfrog
    {
        let path = "./data/leapfrog.parquet";

        let mut system = SpringSystem::lf_new(spring_constant, position, time, mass);
        for _ in 0..steps {
            system.lf_step();
        }

        system.write_to_file(path)?;
    }

    // Runge Kutta 4th Order
    {
        let path = "./data/classical_runge_kutta.parquet";

        let mut system = SpringSystem::new(spring_constant, position, time, mass);
        for _ in 0..steps {
            system.rk4_step();
        }

        system.write_to_file(path)?;
    }

    // Euler
    {
        let path = "./data/euler.parquet";

        let mut system = SpringSystem::new(spring_constant, position, time, mass);
        for _ in 0..steps {
            system.euler_step();
        }

        system.write_to_file(path)?;
    }

    // Euler-Cromer
    {
        let path = "./data/euler_cromer.parquet";

        let mut system = SpringSystem::new(spring_constant, position, time, mass);
        for _ in 0..steps {
            system.ec_step();
        }

        system.write_to_file(path)?;
    }

    // Heun's
    {
        let path = "./data/heun.parquet";

        let mut system = SpringSystem::new(spring_constant, position, time, mass);
        for _ in 0..steps {
            system.heun_step();
        }

        system.write_to_file(path)?;
    }

    Ok(())
}
