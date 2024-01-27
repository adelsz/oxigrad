use std::collections::HashSet;
use std::ops::{Deref, DerefMut};
use std::rc::Rc;
use crate::value_types::input::InputValue;

mod value_types {
    pub mod add;
    pub mod multiply;
    pub mod input;
    pub mod tanh;
}


pub trait DynamicValue {
    fn value(&self) -> f32;
    fn grad(&self) -> f32;
    fn back(&self);
    fn add_grad(&self, grad: f32);
    fn node(&self) -> Vec<Value>;
}
struct Value(Rc<dyn DynamicValue>);

impl Deref for Value {
    type Target = Rc<dyn DynamicValue>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Clone for Value {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl Value {
    fn new(value: f32) -> Self {
        Self(Rc::new(InputValue::new(value)))
    }
}


fn backprop(val: &Value) {
    val.add_grad(1.0);
    val.back();
    let mut queue = vec![val.clone()];
    let mut visited = HashSet::new();
    while let Some(ref node) = queue.pop() {
        for p in node.node() {
            if visited.insert(&*p.0 as *const dyn DynamicValue) {
                p.back();
                queue.push(p.clone());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::value_types::tanh::{tanh};
    use super::*;

    #[test]
    fn mul() {
        let a = Value::new(2.0);
        let b = Value::new(3.0);
        let c = &a * &b;
        assert_eq!(c.value(), 6.0);
        let d = &c * &a;
        assert_eq!(d.value(), 12.0);
        d.add_grad(1.0);
        d.back();
        c.back();
        assert_eq!(a.grad(), 12.0);
    }

    #[test]
    fn internal_loop() {
        let a = Value::new(2.0);
        let b = Value::new(1.0);
        let x = &a * &b;
        let y = &x + &a;
        let w = &x + &b;
        let z = &y * &w;
        z.add_grad(1.0);
        z.back();
        w.back();
        y.back();
        x.back();
        // z = (a * b + a) * (a * b + b) = a^2 * b^2 + a^2 * b + a * b^2 + a * b
        // dz/da = 2ab^2 + 2ab + b^2 + b = 2*2*1 + 2*2 + 1 + 1 = 10
        assert_eq!(a.grad(), 10.0);
    }

    #[test]
    fn backprop_test() {
        let a = Value::new(2.0);
        let b = Value::new(1.0);
        let x = &a * &b;
        let y = &x + &a;
        let w = &x + &b;
        let z = &y * &w;
        backprop(&z);
        // z = (a * b + a) * (a * b + b) = a^2 * b^2 + a^2 * b + a * b^2 + a * b
        // dz/da = 2ab^2 + 2ab + b^2 + b = 2*2*1 + 2*2 + 1 + 1 = 10
        assert_eq!(a.grad(), 10.0);
    }

    #[test]
    fn tanh_test() {
        let a = Value::new(2.0);
        let v = tanh(&a);
        assert_eq!(v.value(), 0.9640276);
        backprop(&v);
        assert_eq!(a.grad(), 0.070650816);
    }
}