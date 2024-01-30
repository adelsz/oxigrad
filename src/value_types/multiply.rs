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
        self.value.get()
    }
    fn set_value(&self, _: f32) {
        panic!("Cannot set value of a dynamic node")
    }
    fn forward(&self)  {
        let operands = self.operands.borrow();
        let (a, b) = operands.deref();
        a.forward();
        b.forward();
        self.value.set(a.value() * b.value());
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
        if Rc::ptr_eq(&a.0, &b.0) {
            a_grad.set(a_grad.get() + grad * 2.0 * a.value());
            return;
        }
        a_grad.set(a_grad.get() + grad * b.value());
        b_grad.set(b_grad.get() + grad * a.value());
    }

    fn dependencies(&self) -> Vec<Value> {
        let operands = self.operands.borrow();
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
