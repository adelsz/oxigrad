use crate::Value;

trait ValueOp {
    fn value(&self) -> f32;
    fn back(&self, grad: f32);
}

struct AddOperation {
    a: Value,
    b: Value,
}

impl AddOperation {
    fn new(a: Value, b: Value) -> Self {
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
}

struct MulOperation {
    a: Value,
    b: Value,
}

impl MulOperation {
    fn new(a: Value, b: Value) -> Self {
        Self { a, b }
    }
}

impl ValueOp for MulOperation {
    fn value(&self) -> f32 {
        self.a.value() * self.b.value()
    }
    fn back(&self, grad: f32) {
        self.a.0.borrow_mut().grad += grad * self.b.value();
        self.b.0.borrow_mut().grad += grad * self.a.value();
    }
}

struct IdOperation {
    a: Value,
}

impl IdOperation {
    fn new(a: Value) -> Self {
        Self { a }
    }
}

impl ValueOp for IdOperation {
    fn value(&self) -> f32 {
        self.a.value()
    }
    fn back(&self, grad: f32) {
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
}