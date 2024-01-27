use std::cell::RefCell;
use crate::{DynamicValue, Value};

pub struct InputValue {
    value: RefCell<f32>,
    grad: RefCell<f32>,
}

impl InputValue {
    pub fn new(value: f32) -> Self {
        Self { value: RefCell::new(value), grad: RefCell::new(0.0) }
    }
    fn set_value(&self, value: f32) {
        *self.value.borrow_mut() = value;
    }

}

impl DynamicValue for InputValue {
    fn value(&self) -> f32 {
        *self.value.borrow()
    }

    fn back(&self) { }

    fn add_grad(&self, grad: f32) {
        *self.grad.borrow_mut() += grad;
    }

    fn grad(&self) -> f32 {
        *self.grad.borrow()
    }

    fn node(&self) -> Vec<Value> {
        vec![]
    }
}
