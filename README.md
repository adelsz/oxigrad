# Oxigrad - toy automatic differentiation engine in Rust

This is a self-contained toy automatic differentiation engine written in Rust.  
Built for learning purposes and partially inspired by [micrograd](https://github.com/karpathy/micrograd).  
API is focused on ease of use and educational clarity, performance was not a priority.

## Usage

```rust
    #[test]
    fn backprop_test() {
        let a = Value::new(2.0);
        let b = Value::new(1.0);
    
        // z = (a * b + a) * (a * b + b) = a^2 * b^2 + a^2 * b + a * b^2 + a * b
        let z = (&a * &b + &a) * (&a * &b + &b);
    
        backprop(&z);
    
        // dz/da = 2ab^2 + 2ab + b^2 + b = 2*2*1 + 2*2 + 1 + 1 = 10
        assert_eq!(a.grad().get(), 10.0);
    }
```

## Basic neural network training

There is a basic neural network training example in `src/neuron.rs` and plots the training progress in a GIF file.

<img src="./test.gif" width=600>
