use std::cell::RefCell;
use std::ops;
use std::rc::Rc;

struct Value {
    value: f32,
    grad: f32,
}

struct Val(Rc<RefCell<Value>>);

impl Clone for Val {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl Copy for Val {}

impl Val {
    fn new(value: f32) -> Self {
        Self(Rc::new(RefCell::new(Value {
            value,
            grad: 0.0,
        })))
    }
}

impl ops::Add for Val {
    type Output = Val;

    fn add(mut self, mut rhs: Self) -> Self::Output {
        let mut a = self.0.borrow_mut();
        let mut b = rhs.0.borrow_mut();
        a.grad += 1.0;
        b.grad += 1.0;
        Val::new(a.value + b.value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sum() {
        let a = Val::new(2.0);
        let b =  Val::new(3.0);
        let result = a.clone() + b.clone();
        assert_eq!(result.0.borrow().value, 5.0);
        assert_eq!(a.0.borrow().value, 5.0);
    }
}
