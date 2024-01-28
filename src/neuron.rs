use crate::Value;

struct Neuron {
    weights: Vec<Value>,
    bias: Value,
    output: Value,
}

fn new_neuron(inputs: &[&Value], activation: fn(&Value) -> Value) -> Neuron {
    let weights: Vec<_> = inputs.iter().map(|_| Value::new(0.0)).collect();
    let bias = Value::new(0.0);
    let result_value = weights.iter().zip(inputs.iter()).fold(bias.clone(), |ref acc, (w, i)| acc + &(w * i));
    let output = activation(&result_value);
    Neuron { weights: weights.clone(), bias: bias.clone(), output }
}


#[cfg(test)]
mod tests {
    use crate::Value;
    use crate::value_types::input::InputValue;
    use crate::value_types::tanh::tanh;
    use super::{Neuron, new_neuron};

    fn gen_test_data() -> Vec<(f32, f32, bool)> {
        let rng = rand::thread_rng();

        todo!()

    }

    #[test]
    fn test_neuron() {
        let x = Value::new(1.0);
        let y = Value::new(2.0);
        let inputs = vec![&x, &y];
        let n1 = new_neuron(&inputs, tanh);
        let n2 = new_neuron(&inputs, tanh);
        let n3 = new_neuron(&inputs, tanh);
        let output = new_neuron(&[&n1.output, &n2.output, &n3.output], tanh);
    }
}

// tanh(x1*w1 + x2*w2 + b)
// tanh(x1 + x2)
// f(x1, x2) = x1 + x2