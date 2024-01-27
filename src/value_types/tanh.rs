use std::cell::RefCell;
use std::ops::DerefMut;
use std::rc::Rc;
use crate::{DynamicValue, Value};

pub struct TanhValue {
    operand: RefCell<Value>,
    grad: RefCell<f32>,
    value: RefCell<f32>,
}

impl TanhValue {
    pub fn new(a: &Value) -> Self {
        Self { operand: RefCell::new(a.clone()), grad: RefCell::new(0.0), value: RefCell::new(a.value().tanh())}
    }
}

impl DynamicValue for TanhValue {
    fn value(&self) -> f32 {
        *self.value.borrow()
    }

    fn grad(&self) -> f32 {
        *self.grad.borrow()
    }

    fn back(&self) {
        let grad = self.grad.borrow().clone();
        let mut operand = self.operand.borrow_mut();
        let a = operand.deref_mut();
        a.add_grad(grad * (1.0 - self.value().powi(2)));
    }

    fn reset_grad(&self) {
        *self.grad.borrow_mut() = 0.0;
    }

    fn add_grad(&self, grad: f32) {
        *self.grad.borrow_mut() += grad;
    }

    fn node(&self) -> Vec<Value> {
        let operand = self.operand.borrow();
        vec![operand.clone()]
    }
}

pub fn tanh(a: &Value) -> Value {
    Value(Rc::new(TanhValue::new(a)))
}
