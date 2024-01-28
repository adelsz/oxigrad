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
    use rand::distributions::Distribution;
    use rand::SeedableRng;
    use crate::{backprop, Value};
    use crate::value_types::input::InputValue;
    use crate::value_types::tanh::tanh;
    use super::{Neuron, new_neuron};

    fn gen_test_data() -> Vec<(f32, f32, bool)> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(1234);
        let dist = rand::distributions::Uniform::new(-1.0, 1.0);
        let sample_count = 10000;
        let mut result = Vec::with_capacity(sample_count);
        for _ in 0..sample_count {
            let x = dist.sample(&mut rng);
            let y = dist.sample(&mut rng);
            let value = (x, y, x*x + y*y < 0.8);
            result.push(value);
        }
        result
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

        let expected = Value::new(0.0);
        let loss = &(&output.output - &expected) * &(&output.output - &expected);

        let mut test_data = gen_test_data();
        let mut epoch = 0;
        let batch_size = 100;

        todo!();
        for batch in test_data.chunks(batch_size) {
            // epoch += 1;
            // let mut error = 0.0;
            // for (xv, yv, expectedv) in batch {
            //     x.value().set(*x);
            //     y.value().set(*y);
            //     expected.value().set(if *expectedv { 1.0 } else { 0.0 });
            //
            //     forward(loss);
            //     backprop(loss);
            // }
            // println!("Epoch: {}, Error: {}", epoch, error);
        }

    }
}

// tanh(x1*w1 + x2*w2 + b)
// tanh(x1 + x2)
// f(x1, x2) = x1 + x2