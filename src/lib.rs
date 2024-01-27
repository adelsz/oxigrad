mod value_ops;

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::ops;
use std::ops::Deref;
use std::rc::Rc;

struct InnerValue {
    value: f32,
    grad: f32,
    op: Operation,
}

struct Value(Rc<RefCell<InnerValue>>);

impl Clone for Value {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

#[derive(Clone)]
enum Operation {
    None,
    Add(Value, Value),
    Mul(Value, Value),
}

impl Value {
    fn new(value: f32) -> Self {
        Self(Rc::new(RefCell::new(InnerValue {
            value,
            grad: 0.0,
            op: Operation::None,
        })))
    }
    fn new_op(value: f32, op: Operation) -> Self {
        Self(Rc::new(RefCell::new(InnerValue {
            value,
            grad: 0.0,
            op,
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
            let op = node.0.borrow().op.clone();
            match op {
                Operation::Add(ref a, ref b) | Operation::Mul(ref a, ref b) => {
                    if visited.insert(&*a.0 as *const RefCell<InnerValue>) {
                        a.back();
                        queue.push(a.clone());
                    }
                    if visited.insert(&*b.0 as *const RefCell<InnerValue>) {
                        b.back();
                        queue.push(b.clone());
                    }
                }
                _ => {}
            }
        }
    }
    fn back(&self) {
        let mut op = self.0.borrow_mut();
        if let Operation::None = op.op {
            return;
        }
        let grad = op.grad;
        match op.op {
            Operation::Add(ref a, ref b) => {
                a.0.borrow_mut().grad += grad;
                b.0.borrow_mut().grad += grad;
            }
            Operation::Mul(ref a, ref b) => {
                if Rc::ptr_eq(&a.0, &b.0) {
                    let value = a.0.borrow().value;
                    a.0.borrow_mut().grad += 2.0*grad * value;
                } else {
                    a.0.borrow_mut().grad += grad * b.0.borrow().value;
                    b.0.borrow_mut().grad += grad * a.0.borrow().value;
                }
            }
            Operation::None => {}
        }
    }
}

impl ops::Add for Value {
    type Output = Value;

    fn add(mut self, mut rhs: Self) -> Self::Output {
        Value::new_op(self.0.borrow().value + rhs.0.borrow().value, Operation::Add(self.clone(), rhs.clone()))
    }
}

impl ops::Mul for Value {
    type Output = Value;

    fn mul(mut self, mut rhs: Self) -> Self::Output {
        Value::new_op(self.0.borrow().value * rhs.0.borrow().value, Operation::Mul(self.clone(), rhs.clone()))
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
