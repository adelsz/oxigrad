use std::cell::{Cell, RefCell};
use std::ops::DerefMut;
use std::rc::Rc;
use crate::{DynamicValue, Value};

pub struct TanhValue {
    operand: RefCell<Value>,
    grad: Cell<f32>,
    value: Cell<f32>,
}

impl TanhValue {
    pub fn new(a: &Value) -> Self {
        Self { operand: RefCell::new(a.clone()), grad: Cell::new(0.0), value: Cell::new(a.value().tanh())}
    }
}

impl DynamicValue for TanhValue {
    fn value(&self) -> f32 {
       self.value.get()
    }

    fn grad(&self) -> f32 {
        self.grad.get()
    }

    fn back(&self) {
        let grad = self.grad.get();
        let mut operand = self.operand.borrow_mut();
        let a = operand.deref_mut();
        a.add_grad(grad * (1.0 - self.value().powi(2)));
    }

    fn reset_grad(&self) {
        self.grad.set(0.0)
    }

    fn add_grad(&self, grad: f32) {
        self.grad.set(self.grad.get() + grad);
    }

    fn node(&self) -> Vec<Value> {
        let operand = self.operand.borrow();
        vec![operand.clone()]
    }
}

pub fn tanh(a: &Value) -> Value {
    Value(Rc::new(TanhValue::new(a)))
}
