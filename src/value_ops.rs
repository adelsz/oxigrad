use std::rc::Rc;
use crate::Value;

pub trait ValueOp {
    fn value(&self) -> f32;
    fn back(&self, grad: f32);
    fn node(&self) -> Vec<Value>;
}

pub struct AddOperation {
    a: Value,
    b: Value,
}

impl AddOperation {
    pub fn new(a: Value, b: Value) -> Self {
        Self { a, b }
    }
}

impl ValueOp for AddOperation {
    fn value(&self) -> f32 {
        self.a.value() + self.b.value()
    }
    fn back(&self, grad: f32) {
        self.a.0.borrow_mut().grad += grad;
        self.b.0.borrow_mut().grad += grad;
    }
    fn node(&self) -> Vec<Value> {
        vec![self.a.clone(), self.b.clone()]
    }
}

pub struct MulOperation {
    a: Value,
    b: Value,
}

impl MulOperation {
    pub fn new(a: Value, b: Value) -> Self {
        Self { a, b }
    }
}

impl ValueOp for MulOperation {
    fn value(&self) -> f32 {
        self.a.value() * self.b.value()
    }
    fn back(&self, grad: f32) {
        if Rc::ptr_eq(&self.a.0, &self.b.0) {
            self.a.0.borrow_mut().grad += grad * 2.0 * self.a.value();
            return;
        }
        self.a.0.borrow_mut().grad += grad * self.b.value();
        self.b.0.borrow_mut().grad += grad * self.a.value();
    }
    fn node(&self) -> Vec<Value> {
        vec![self.a.clone(), self.b.clone()]
    }
}

pub(crate) struct NoneOperation {
}

impl NoneOperation {
    pub(crate) fn new() -> Self {
        Self { }
    }
}

impl ValueOp for NoneOperation {
    fn value(&self) -> f32 {
        panic!("None operation has no value and should not be called")
    }
    fn back(&self, grad: f32) {
    }
    fn node(&self) -> Vec<Value> {
        vec![]
    }
}

struct ExpOperation {
    a: Value,
}

impl ExpOperation {
    fn new(a: Value) -> Self {
        Self { a }
    }
}

impl ValueOp for ExpOperation {
    fn value(&self) -> f32 {
        self.a.value().exp()
    }
    fn back(&self, grad: f32) {
        self.a.0.borrow_mut().grad += grad * self.value();
    }
    fn node(&self) -> Vec<Value> {
        vec![self.a.clone()]
    }
}