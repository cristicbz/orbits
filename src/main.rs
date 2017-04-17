extern crate rand;

use std::ops::{Add, Sub, Mul, Div, AddAssign, SubAssign, MulAssign, DivAssign, Neg};
use rand::{XorShiftRng, Rng, SeedableRng, Rand};
use rand::distributions::normal::StandardNormal;
use std::fmt::{Formatter, Result as FmtResult, Display};


const PLANET_RADIUS: Scalar = 6.371e6;
const GRAVITATIONAL_PARAMETER: Scalar = 3.9860044181e14;
const TIMESTEP: Scalar = 1e-4;
const DESIRED_ORBIT: Scalar = PLANET_RADIUS + 160e3;
const MAX_ERROR: Scalar = 1.0e3;

const SHIP_MASS: Scalar = 500.0;
const INITIAL_FUEL: Scalar = 50.0;
const MAX_FLOW_RATE: Scalar = 100.0 * 1e-3;
const MAX_THRUST: Scalar = 270.0;
const MIN_THRUST: Scalar = 180.0;

const CROSS_SECTION_RADIUS: Scalar = 1.5;
const CROSS_SECTION_AREA: Scalar = 3.14159 * CROSS_SECTION_RADIUS * CROSS_SECTION_RADIUS;
const SPHERE_DRAG_COEFFICIENT: Scalar = 0.5;

const RANDOM_MOMENTUM_FREQUENCY: Scalar = 1.0 * 60.0 * 60.0; // Every hour get hit by something.
const RANDOM_MOMENTUM_MEAN: Scalar = 3e2;
const RANDOM_MOMENTUM_STD: Scalar = 1e1;


const NUM_TIMESTEPS: usize = 1_000_000_000;

const SEEDS: [u32; 4] = [1, 2, 3, 4];


// https://en.wikipedia.org/wiki/Atmospheric_pressure
const AIR_GAS_CONSTANT: Scalar = 287.0;

const DENSITY_A: Scalar = 1.225;
const DENSITY_K: Scalar = 1.38785564661078223780e-4;

const PRESSURE_A: Scalar = 1.013e5;
const PRESSURE_K: Scalar = 1.42881643880405620386e-4;


#[derive(Copy, Clone, Debug)]
pub struct Atmosphere {
    pressure: Scalar,
    temperature: Scalar,
    density: Scalar,
}

impl Atmosphere {
    pub fn at(distance: Scalar) -> Atmosphere {
        let height = distance - PLANET_RADIUS;
        let pressure = PRESSURE_A * (-PRESSURE_K * height).exp();
        let density = DENSITY_A * (-DENSITY_K * height).exp();
        let temperature = pressure / (AIR_GAS_CONSTANT * density);

        Atmosphere {
            pressure: pressure,
            temperature: temperature,
            density: density,
        }
    }
}


#[derive(Copy, Clone, Debug)]
pub struct ShipState {
    pub position: Vec3,
    pub velocity: Vec3,
    pub fuel: Scalar,
    pub time: Scalar,
}

impl ShipState {
    pub fn total_mass(&self) -> Scalar {
        SHIP_MASS + self.fuel
    }
}

pub trait Controller {
    fn requested_thrust(&mut self, atmosphere: &Atmosphere, state: &ShipState) -> Vec3;
}

pub struct NoopController;
impl Controller for NoopController {
    fn requested_thrust(&mut self, _atmosphere: &Atmosphere, _state: &ShipState) -> Vec3 {
        Vec3(0.0, 0.0, 0.0)
    }
}

pub struct PidController {
    p: Scalar,
    i: Scalar,
    d: Scalar,

    last_error: Vec3,
    error_integral: Vec3,
}

impl PidController {
    pub fn new(p: Scalar, i: Scalar, d: Scalar) -> Self {
        PidController {
            p: p,
            i: i,
            d: d,
            last_error: Vec3::default(),
            error_integral: Vec3::default(),
        }
    }
}

impl Controller for PidController {
    fn requested_thrust(&mut self, _atmosphere: &Atmosphere, state: &ShipState) -> Vec3 {
        let closest_point_on_orbit = Vec3(state.position.0, 0.0, state.position.2).normalised() *
                                     DESIRED_ORBIT;
        let error = closest_point_on_orbit - state.position;

        let thrust = error * self.p + self.error_integral * self.i +
                     (self.last_error - error) * self.d;
        self.last_error = error;
        self.error_integral += error;

        thrust
    }
}

#[derive(Debug)]
pub struct End {
    pub crashed: bool,
    pub state: ShipState,
    pub time_in_orbit: Scalar,
    pub num_timesteps: usize,
    pub distance: Scalar,
    pub travelled: Scalar,
}

fn simulate<C: Controller>(index: usize, _seed: u64, mut controller: C) -> End {
    let mut rng = XorShiftRng::from_seed(SEEDS);
    let timestep_flow_rate_per_newton = MAX_FLOW_RATE / MAX_THRUST * TIMESTEP;

    let mut state = ShipState {
        position: Vec3(0.0, 0.0, DESIRED_ORBIT),
        velocity: Vec3((GRAVITATIONAL_PARAMETER / DESIRED_ORBIT).sqrt(), 0.0, 0.0),
        fuel: INITIAL_FUEL,
        time: 0.0,
    };
    let mut travelled = 0.0;
    let mut time_in_orbit = 0.0;

    let mut num_hits = 0;
    let mut last_momentum = 0.0;
    let mut last_momentum_theta: Scalar = 0.0;

    for i_timestep in 1..NUM_TIMESTEPS + 1 {
        state.time += TIMESTEP * 0.5;
        state.position += state.velocity * (TIMESTEP * 0.5);


        let speed_squared = state.velocity.norm_squared();
        let speed = speed_squared.sqrt();
        travelled += speed * TIMESTEP;

        let distance_squared = state.position.norm_squared();
        let distance = distance_squared.sqrt();
        let atmosphere = Atmosphere::at(distance);

        let closest_point_on_orbit = Vec3(state.position.0, 0.0, state.position.2).normalised() *
                                     DESIRED_ORBIT;
        let error_squared = (closest_point_on_orbit - state.position).norm_squared();
        if error_squared <= MAX_ERROR * MAX_ERROR {
            time_in_orbit += TIMESTEP;
        }

        let crashed = distance <= PLANET_RADIUS;
        if crashed || (i_timestep % 1_000_000 == 0) {
            let mass = state.total_mass();
            let energy = (state.velocity.norm_squared() * 0.5 -
                          GRAVITATIONAL_PARAMETER / distance) * mass;
            println!("=== Status: {}: {:9}: ===\n\
                     Time: {:.2}\n\
                     Energy: {:.5e}\n\
                     Time in orbit: {:.2} ({:.2}% of total)\n\
                     Error: {:.5}km\n\
                     Altitude: {:.5}km\n\
                     Travelled: {:.5}km\n\
                     Speed: {:.5}km/s\n\
                     Atmosphere: {:.5e}Pa, {:.2}C, {:.5e}kg/m^3\n\
                     Fuel: {:.2}kg\n\
                     Hits: {} (last: {:.2}kg*m/s, theta: {:.2} deg)\n\
                     Position: {:?}\n\
                     Velocity: {:?}\n\n",
                     index,
                     i_timestep,
                     Time(state.time),
                     energy,
                     Time(time_in_orbit),
                     time_in_orbit / state.time * 100.0,
                     error_squared.sqrt() / 1000.0,
                     (distance - PLANET_RADIUS).max(0.0) / 1000.0,
                     travelled / 1000.0,
                     state.velocity.norm() / 1000.0,
                     atmosphere.pressure,
                     atmosphere.temperature - 273.15,
                     atmosphere.density,
                     state.fuel,
                     num_hits,
                     last_momentum,
                     last_momentum_theta.to_degrees(),
                     state.position,
                     state.velocity);
        }

        if crashed {
            let end = End {
                crashed: true,
                state: state,
                time_in_orbit: time_in_orbit,
                num_timesteps: i_timestep,
                distance: distance,
                travelled: travelled,
            };
            //println!("=== {}: {:9}: Crashed! ===\n{:#?}", index, i_timestep, end);
            return end;
        }

        if rng.gen::<f64>() <= (TIMESTEP / RANDOM_MOMENTUM_FREQUENCY) {
            let momentum = rng.gen::<Vec3>().normalised() *
                           (rng.gen::<StandardNormal>().0 * RANDOM_MOMENTUM_STD +
                            RANDOM_MOMENTUM_MEAN);
            last_momentum = momentum.norm();
            let dot = momentum.dot(&state.velocity);
            last_momentum_theta = (dot / (last_momentum * speed)).acos() * dot.signum();
            num_hits += 1;
            state.velocity += momentum / SHIP_MASS;
        }

        let mut acceleration = state.position *
                               (-GRAVITATIONAL_PARAMETER / (distance_squared * distance));
        let mut forces = -SPHERE_DRAG_COEFFICIENT * CROSS_SECTION_AREA * atmosphere.density *
                         speed * state.velocity;
        let mut mass = state.total_mass();
        if state.fuel > 0.0 {
            let requested_thrust = controller.requested_thrust(&atmosphere, &state);
            let requested_thrust_norm_squared = requested_thrust.norm_squared();
            if requested_thrust_norm_squared > 0.0 &&
               requested_thrust_norm_squared >= MIN_THRUST * MIN_THRUST {
                let requested_thrust_norm = requested_thrust_norm_squared.sqrt();
                let max_thrust = MAX_THRUST.min(state.fuel / timestep_flow_rate_per_newton);
                let (thrust, thrust_norm) = if requested_thrust_norm > max_thrust {
                    (requested_thrust * (max_thrust / requested_thrust_norm), max_thrust)
                } else {
                    (requested_thrust, requested_thrust_norm)
                };

                let fuel_consumption = thrust_norm * timestep_flow_rate_per_newton;

                mass -= fuel_consumption / 2.0;
                forces += thrust;

                state.fuel -= fuel_consumption;
                if state.fuel < 0.0 {
                    state.fuel = 0.0;
                }
            }
        }
        acceleration += forces / mass;
        state.velocity += acceleration * TIMESTEP;
        state.position += state.velocity * (TIMESTEP * 0.5);
        state.time += TIMESTEP * 0.5;
    }

    let end = End {
        crashed: false,
        distance: state.position.norm(),
        state: state,
        time_in_orbit: time_in_orbit,
        num_timesteps: NUM_TIMESTEPS,
        travelled: travelled,
    };
    //println!("=== {}: {:9}: Done! ===\n{:#?}", index, NUM_TIMESTEPS, end);
    end
}

fn main() {
    simulate(0, 0, PidController::new(0.2, 0.0, 0.0125));
}

pub struct Time(pub Scalar);

impl Display for Time {
    fn fmt(&self, fmt: &mut Formatter) -> FmtResult {
        const MINUTE: Scalar = 60.0;
        const HOUR: Scalar = 60.0 * MINUTE;
        const DAY: Scalar = 24.0 * HOUR;

        let mut seconds = self.0;
        let days = (seconds / DAY).floor();
        seconds -= days * DAY;
        let hours = (seconds / HOUR).floor();
        seconds -= hours * HOUR;
        let minutes = (seconds / MINUTE).floor();
        seconds -= minutes * MINUTE;

        write!(fmt, "{}d {}h {}m {:.2}s", days, hours, minutes, seconds)
    }
}

type Scalar = f64;

#[derive(Copy, Clone, Debug, PartialEq, PartialOrd, Default)]
pub struct Vec3(pub Scalar, pub Scalar, pub Scalar);

impl Vec3 {
    pub fn dot(&self, rhs: &Vec3) -> Scalar {
        self.0 * rhs.0 + self.1 * rhs.1 + self.2 * rhs.2
    }

    pub fn norm_squared(&self) -> Scalar {
        self.dot(self)
    }

    pub fn norm(&self) -> Scalar {
        self.norm_squared().sqrt()
    }

    pub fn normalised(&self) -> Vec3 {
        let norm_squared = self.norm_squared();
        if norm_squared > 0.0 {
            self / norm_squared.sqrt()
        } else {
            *self
        }
    }
}

impl Rand for Vec3 {
    fn rand<R: Rng>(rng: &mut R) -> Self {
        Vec3(rng.gen::<StandardNormal>().0,
             rng.gen::<StandardNormal>().0,
             rng.gen::<StandardNormal>().0)
    }
}

impl Neg for Vec3 {
    type Output = Vec3;
    fn neg(self) -> Vec3 {
        Vec3(-self.0, -self.1, -self.2)
    }
}

impl<'a> Neg for &'a Vec3 {
    type Output = Vec3;
    fn neg(self) -> Vec3 {
        -(*self)
    }
}

macro_rules! impl_ops {
    ($([$op:ident :: $fun:ident, $op_assign:ident :: $fun_assign:ident],)+) => {
        $(
            impl $op for Vec3 {
                type Output = Vec3;

                fn $fun(self, rhs: Vec3) -> Vec3 {
                    Vec3($op::$fun(self.0, rhs.0),
                    $op::$fun(self.1, rhs.1),
                    $op::$fun(self.2, rhs.2))
                }
            }

            impl<'a> $op<Vec3> for &'a Vec3 {
                type Output = Vec3;

                fn $fun(self, rhs: Vec3) -> Vec3 {
                    $op::$fun(*self, rhs)
                }
            }

            impl<'a> $op<&'a Vec3> for Vec3 {
                type Output = Vec3;

                fn $fun(self, rhs: &'a Vec3) -> Vec3 {
                    $op::$fun(self, *rhs)
                }
            }

            impl<'a, 'b> $op<&'a Vec3> for &'b Vec3 {
                type Output = Vec3;

                fn $fun(self, rhs: &'a Vec3) -> Vec3 {
                    $op::$fun(*self, *rhs)
                }
            }

            impl $op<Scalar> for Vec3 {
                type Output = Vec3;

                fn $fun(self, rhs: Scalar) -> Vec3 {
                    Vec3($op::$fun(self.0, rhs),
                    $op::$fun(self.1, rhs),
                    $op::$fun(self.2, rhs))
                }
            }

            impl<'a> $op<Scalar> for &'a Vec3 {
                type Output = Vec3;

                fn $fun(self, rhs: Scalar) -> Vec3 {
                    $op::$fun(*self, rhs)
                }
            }

            impl<'a> $op<&'a Scalar> for Vec3 {
                type Output = Vec3;

                fn $fun(self, rhs: &'a Scalar) -> Vec3 {
                    $op::$fun(self, *rhs)
                }
            }

            impl<'a, 'b> $op<&'a Scalar> for &'b Vec3 {
                type Output = Vec3;

                fn $fun(self, rhs: &'a Scalar) -> Vec3 {
                    $op::$fun(*self, *rhs)
                }
            }

            impl $op<Vec3> for Scalar {
                type Output = Vec3;

                fn $fun(self, rhs: Vec3) -> Vec3 {
                    Vec3($op::$fun(self, rhs.0),
                    $op::$fun(self, rhs.1),
                    $op::$fun(self, rhs.2))
                }
            }

            impl<'a> $op<Vec3> for &'a Scalar {
                type Output = Vec3;

                fn $fun(self, rhs: Vec3) -> Vec3 {
                    $op::$fun(*self, rhs)
                }
            }

            impl<'a> $op<&'a Vec3> for Scalar {
                type Output = Vec3;

                fn $fun(self, rhs: &'a Vec3) -> Vec3 {
                    $op::$fun(self, *rhs)
                }
            }

            impl<'a, 'b> $op<&'a Vec3> for &'b Scalar {
                type Output = Vec3;

                fn $fun(self, rhs: &'a Vec3) -> Vec3 {
                    $op::$fun(*self, *rhs)
                }
            }

            impl $op_assign for Vec3 {
                fn $fun_assign(&mut self, rhs: Vec3) {
                    $op_assign::$fun_assign(&mut self.0, rhs.0);
                    $op_assign::$fun_assign(&mut self.1, rhs.1);
                    $op_assign::$fun_assign(&mut self.2, rhs.2);
                }
            }

            impl<'a> $op_assign<&'a Vec3> for Vec3 {
                fn $fun_assign(&mut self, rhs: &'a Vec3) {
                    $op_assign::$fun_assign(self, *rhs);
                }
            }

            impl $op_assign<Scalar> for Vec3 {
                fn $fun_assign(&mut self, rhs: Scalar) {
                    $op_assign::$fun_assign(&mut self.0, rhs);
                    $op_assign::$fun_assign(&mut self.1, rhs);
                    $op_assign::$fun_assign(&mut self.2, rhs);
                }
            }

            impl<'a> $op_assign<&'a Scalar> for Vec3 {
                fn $fun_assign(&mut self, rhs: &'a Scalar) {
                    $op_assign::$fun_assign(self, *rhs);
                }
            }
            )+
    }
}
impl_ops! {
    [Add::add, AddAssign::add_assign],
    [Sub::sub, SubAssign::sub_assign],
    [Mul::mul, MulAssign::mul_assign],
    [Div::div, DivAssign::div_assign],
}
