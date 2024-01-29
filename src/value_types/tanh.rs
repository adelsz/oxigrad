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
    fn set_value(&self, value: f32) {
        panic!("Cannot set value of a dynamic node")
    }

    fn forward(&self) {
        let operand = self.operand.borrow();
        operand.forward();
        self.value.set(operand.value().tanh());
    }

    fn grad(&self) -> &Cell<f32> {
        &self.grad
    }

    fn back(&self) {
        let grad = self.grad.get();
        let mut operand = self.operand.borrow_mut();
        let a = operand.deref_mut();
        let new_value = 1.0 - a.value().tanh().powi(2);
        a.grad().set(a.grad().get() + grad*(new_value));
    }

    fn dependencies(&self) -> Vec<Value> {
        let operand = self.operand.borrow();
        vec![operand.clone()]
    }
}

pub fn tanh(a: &Value) -> Value {
    Value(Rc::new(TanhValue::new(a)))
}
