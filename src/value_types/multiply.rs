use std::ops;
use std::rc::Rc;
use std::cell::{Cell, RefCell};
use std::ops::{Deref, DerefMut};
use crate::{DynamicValue, Value};

pub struct MulValue {
    operands: RefCell<(Value, Value)>,
    grad: Cell<f32>,
    value: Cell<f32>,
}

impl MulValue {
    pub fn new(a: Value, b: Value) -> Self {
        Self { operands: RefCell::new((a.clone(), b.clone())), grad: Cell::new(0.0), value: Cell::new(a.value() * b.value())}
    }
}

impl DynamicValue for MulValue {
    fn value(&self) -> f32 {
        let operands = self.operands.borrow();
        let (a, b) = operands.deref();
        a.value() * b.value()
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
        if Rc::ptr_eq(&a.0, &b.0) {
            a.add_grad(grad * 2.0 * a.value());
            return;
        }
        a.add_grad(grad * b.value());
        b.add_grad(grad * a.value());
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

impl ops::Mul for &Value {
    type Output = Value;

    fn mul(self, rhs: Self) -> Self::Output {
        Value(Rc::new(MulValue::new(self.clone(), rhs.clone())))
    }
}
