mod value_ops;

use std::cell::RefCell;
use std::collections::{HashSet};
use std::ops;
use std::ops::Deref;
use std::rc::Rc;
use crate::value_ops::{AddOperation, MulOperation, ValueOp};

struct InnerValue {
    value: f32,
    grad: f32,
    op: Box<dyn ValueOp>,
}

struct Value(Rc<RefCell<InnerValue>>);

impl Clone for Value {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl Value {
    fn new(value: f32) -> Self {
        Self(Rc::new(RefCell::new(InnerValue {
            value,
            grad: 0.0,
            op: Box::new(value_ops::NoneOperation::new()),
        })))
    }
    fn new_op<A: ValueOp + 'static>(value: f32, op: A) -> Self {
        Self(Rc::new(RefCell::new(InnerValue {
            value,
            grad: 0.0,
            op: Box::new(op),
        })))
    }
    fn value(&self) -> f32 {
        self.0.borrow().value
    }
    fn grad(&self) -> f32 {
        self.0.borrow().grad
    }
    fn backprop(&mut self) {
        self.0.borrow_mut().grad = 1.0;
        self.back();
        let mut queue = vec![self.clone()];
        let mut visited = HashSet::new();
        while let Some(ref node) = queue.pop() {
            let op = node.0.borrow();
            for p in op.op.node() {
                if visited.insert(&*p.0 as *const RefCell<InnerValue>) {
                    p.back();
                    queue.push(p.clone());
                }
            }
        }
    }
    fn back(&self) {
        let grad = self.0.borrow().grad;
        self.0.borrow_mut().op.back(grad);
    }
}

impl ops::Add for Value {
    type Output = Value;

    fn add(mut self, mut rhs: Self) -> Self::Output {
        let op = AddOperation::new(self.clone(), rhs.clone());
        Value::new_op(self.0.borrow().value + rhs.0.borrow().value, op)
    }
}

impl ops::Mul for Value {
    type Output = Value;

    fn mul(mut self, mut rhs: Self) -> Self::Output {
        let op = MulOperation::new(self.clone(), rhs.clone());
        Value::new_op(self.0.borrow().value * rhs.0.borrow().value, op)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sum() {
        let a = Value::new(2.0);
        let b =  Value::new(3.0);
        let x = a.clone() + b.clone();
        let y = a.clone();
        let z = x.clone() + y.clone();
        // z = (a + b) + a
        z.0.borrow_mut().grad = 1.0;
        z.back();
        y.back();
        x.back();
        assert_eq!(a.grad(), 2.0);
    }

    #[test]
    fn mul() {
        let a = Value::new(5.0);
        let x = a.clone() * a.clone();
        let y = x.clone() * a.clone();
        y.0.borrow_mut().grad = 1.0;
        y.back();
        assert_eq!(x.grad(), 5.0);
        assert_eq!(a.grad(), 25.0);
        x.back();
        assert_eq!(a.grad(), 75.0);
    }

    #[test]
    fn internal_loop() {
        let a = Value::new(2.0);
        let b = Value::new(1.0);
        let x = a.clone() * b.clone();
        let y = x.clone() + a.clone();
        let w = x.clone() + b.clone();
        let mut z = y.clone() * w.clone();
        z.0.borrow_mut().grad = 1.0;
        z.back();
        w.back();
        y.back();
        x.back();
        // z = (a * b + a) * (a * b + b) = a^2 * b^2 + a^2 * b + a * b^2 + a * b
        // dz/da = 2ab^2 + 2ab + b^2 + b = 2*2*1 + 2*2 + 1 + 1 = 10
        assert_eq!(a.grad(), 10.0);
    }

    #[test]
    fn backprop() {
        let a = Value::new(2.0);
        let b = Value::new(1.0);
        let x = a.clone() * b.clone();
        let y = x.clone() + a.clone();
        let w = x.clone() + b.clone();
        let mut z = y.clone() * w.clone();
        z.backprop();
        assert_eq!(a.grad(), 10.0);
    }
}
