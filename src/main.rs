extern crate rand;

use std::ops::{Add, Sub, Mul, Div, AddAssign, SubAssign, MulAssign, DivAssign, Neg};
use rand::{XorShiftRng, Rng, SeedableRng};


// http://www.space-propulsion.com/spacecraft-propulsion/bipropellant-thrusters/220n-atv-thrusters.html
const PLANET_RADIUS: Scalar = 6.371e6;
const GRAVITATIONAL_PARAMETER: Scalar = 3.9860044181e14;
const TIMESTEP: Scalar = 1e-4;
const DESIRED_ORBIT: Scalar = PLANET_RADIUS + 200e3;

const SHIP_MASS: Scalar = 500.0;
const INITIAL_FUEL: Scalar = 50.0;
const MAX_FLOW_RATE: Scalar = 100.0 * 1e-3;
const MAX_THRUST: Scalar = 270.0;
const MIN_THRUST: Scalar = 0.0; // Actually 180.0

const CROSS_SECTION_RADIUS: Scalar = 1.5;
const CROSS_SECTION_AREA: Scalar = 3.14159 * CROSS_SECTION_RADIUS * CROSS_SECTION_RADIUS;
const SPHERE_DRAG_COEFFICIENT: Scalar = 0.5;

const NUM_TIMESTEPS: usize = 10_000_000_000;

const SEEDS: [u64; 4] = [1, 2, 3, 4];


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

#[derive(Debug)]
pub struct End {
    pub crashed: bool,
    pub state: ShipState,
    pub cumulative_error: Scalar,
    pub num_timesteps: usize,
    pub distance: Scalar,
    pub travelled: Scalar,

}

fn simulate<C: Controller>(index: usize, _seed: u64, mut controller: C) -> End {
    let timestep_flow_rate_per_newton = MAX_FLOW_RATE / MAX_THRUST * TIMESTEP;


    let mut state = ShipState {
        position: Vec3(0.0, 0.0, DESIRED_ORBIT),
        velocity: Vec3((GRAVITATIONAL_PARAMETER / DESIRED_ORBIT).sqrt(), 0.0, 0.0),
        fuel: INITIAL_FUEL,
        time: 0.0
    };
    let mut travelled = 0.0;
    let mut error = 0.0;

    for i_timestep in 1..NUM_TIMESTEPS + 1 {
        state.time += TIMESTEP * 0.5;
        state.position += state.velocity * (TIMESTEP * 0.5);

        let distance_from_orbit =
            (state.position.0 * state.position.0 + state.position.2 * state.position.2).sqrt() -
            DESIRED_ORBIT;
        error += distance_from_orbit * distance_from_orbit * TIMESTEP;

        let speed_squared = state.velocity.norm_squared();
        let speed = speed_squared.sqrt();
        travelled += speed * TIMESTEP;

        let distance_squared = state.position.norm_squared();
        let distance = distance_squared.sqrt();
        let atmosphere = Atmosphere::at(distance);

        let crashed = distance <= PLANET_RADIUS;
        if crashed || (i_timestep % 1_000_000 == 0) {
            let mass = state.total_mass();
            let energy = (state.velocity.norm_squared() * 0.5 -
                          GRAVITATIONAL_PARAMETER / distance) * mass;
            println!("=== Status: {}: {:9}: ===\nEnergy: {:e}\nError: {:e}\nDistance: {:.5}km\n\
                     Travelled: {:.5}km\nSpeed: {:.5}km/s\n{:#?}\n{:#?}",
                     index, i_timestep, energy, error,
                     (distance - PLANET_RADIUS) / 1000.0,
                     travelled / 1000.0, state.velocity.norm() / 1000.0,
                     atmosphere,
                     state);
        }

        if crashed {
            let end = End {
                crashed: true,
                state: state,
                cumulative_error: error,
                num_timesteps: i_timestep,
                distance: distance,
                travelled: travelled,
            };
            println!("=== {}: {:9}: Crashed! ===\n{:#?}", index, i_timestep, end);
            return end;
        }


        let mut acceleration =
            state.position * (-GRAVITATIONAL_PARAMETER / (distance_squared * distance));
        let mut forces =
            -SPHERE_DRAG_COEFFICIENT * CROSS_SECTION_AREA * atmosphere.density * speed *
            state.velocity;
        let mut mass = state.total_mass();
        if state.fuel > 0.0 {
            let requested_thrust = controller.requested_thrust(&atmosphere, &state);
            let requested_thrust_norm_squared = requested_thrust.norm_squared();
            if requested_thrust_norm_squared > 0.0 {
                let requested_thrust_norm = requested_thrust_norm_squared.sqrt();
                let max_thrust = MAX_THRUST.min(state.fuel / timestep_flow_rate_per_newton);
                let (thrust, thrust_norm) = if requested_thrust_norm > max_thrust {
                    (requested_thrust * (max_thrust / requested_thrust_norm),
                     max_thrust)
                } else {
                    (requested_thrust, requested_thrust_norm)
                };

                let fuel_consumption = thrust_norm * timestep_flow_rate_per_newton;

                mass -= fuel_consumption / 2.0;
                forces += thrust;

                state.fuel -= fuel_consumption;
                if state.fuel <= 0.0 {
                    println!("{}: {:9}: Out of fuel! {:?}", index, i_timestep, state);
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
        cumulative_error: error,
        num_timesteps: NUM_TIMESTEPS,
        travelled: travelled,
    };
    println!("=== {}: {:9}: Done! ===\n{:#?}", index, NUM_TIMESTEPS, end);
    end
}

fn main() {
    simulate(0, 0, NoopController);
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
