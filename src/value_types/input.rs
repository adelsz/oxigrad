use std::cell::{Cell};
use crate::{DynamicValue, Value};

pub struct InputValue {
    value: Cell<f32>,
    grad: Cell<f32>,
}

impl InputValue {
    pub fn new(value: f32) -> Self {
        Self { value: Cell::new(value), grad: Cell::new(0.0) }
    }
}

impl DynamicValue for InputValue {
    fn value(&self) -> f32 {
        self.value.get()
    }

    fn set_value(&self, value: f32) {
        self.value.set(value);
    }

    fn forward(&self) {
    }

    fn back(&self) { }

    fn grad(&self) -> &Cell<f32> {
        &self.grad
    }

    fn dependencies(&self) -> Vec<Value> {
        vec![]
    }
}
