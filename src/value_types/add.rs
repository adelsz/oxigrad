use std::cell::{Cell, RefCell};
use std::ops;
use crate::{DynamicValue, Value};
use std::ops::{Deref, DerefMut};
use std::rc::Rc;

pub struct AddValue {
    operands: RefCell<(Value, Value)>,
    grad: Cell<f32>,
    value: Cell<f32>,
}

impl ops::Add for &Value {
    type Output = Value;

    fn add(self, rhs: Self) -> Self::Output {
        Value(Rc::new(AddValue::new(self.clone(), rhs.clone())))
    }
}

impl AddValue {
    pub fn new(a: Value, b: Value) -> Self {
        Self { operands: RefCell::new((a.clone(), b.clone())), grad: Cell::new(0.0), value: Cell::new(a.value() + b.value())}
    }
}

impl DynamicValue for AddValue {
    fn value(&self) -> f32 {
        let operands = self.operands.borrow();
        let (a, b) = operands.deref();
        a.value() + b.value()
    }

    fn grad(&self) -> f32 {
        self.grad.get()
    }

    fn add_grad(&self, grad: f32) {
        self.grad.set(self.grad.get() + grad);
    }

    fn back(&self) {
        let grad = self.grad.get();
        let mut operands = self.operands.borrow_mut();
        let (a, b) = operands.deref_mut();
        a.add_grad(grad);
        b.add_grad(grad);
    }

    fn reset_grad(&self) {
        self.grad.set(0.0)
    }

    fn node(&self) -> Vec<Value> {
        let mut operands = self.operands.borrow();
        let (a, b) = operands.deref();
        vec![a.clone(), b.clone()]
    }
}
