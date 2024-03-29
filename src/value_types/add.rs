use std::borrow::Borrow;
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

impl <X: Borrow<Value>>ops::Add<X> for &Value {
    type Output = Value;

    fn add(self, rhs: X) -> Self::Output {
        Value(Rc::new(AddValue::new(self.clone(), rhs.borrow().clone())))
    }
}

impl <X: Borrow<Value>>ops::Add<X> for Value {
    type Output = Value;

    fn add(self, rhs: X) -> Self::Output {
        &self + rhs.borrow()
    }
}

impl AddValue {
    pub fn new(a: Value, b: Value) -> Self {
        Self { operands: RefCell::new((a.clone(), b.clone())), grad: Cell::new(0.0), value: Cell::new(a.value() + b.value())}
    }
}

impl DynamicValue for AddValue {
    fn value(&self) -> f32 {
        self.value.get()
    }

    fn set_value(&self, _: f32) {
        panic!("Cannot set value of a dynamic node")
    }

    fn forward(&self) {
        let operands = self.operands.borrow();
        let (a, b) = operands.deref();
        a.forward();
        b.forward();
        self.value.set(a.value() + b.value());
    }

    fn grad(&self) -> &Cell<f32> {
        &self.grad
    }

    fn back(&self) {
        let grad = self.grad.get();
        let mut operands = self.operands.borrow_mut();
        let (a, b) = operands.deref_mut();
        let a_grad = a.grad();
        let b_grad = b.grad();
        a_grad.set(a_grad.get() + grad);
        b_grad.set(b_grad.get() + grad);
    }

    fn dependencies(&self) -> Vec<Value> {
        let operands = self.operands.borrow();
        let (a, b) = operands.deref();
        vec![a.clone(), b.clone()]
    }
}
