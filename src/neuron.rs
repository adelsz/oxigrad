use rand::{Rng, RngCore, SeedableRng};
use crate::Value;

struct Neuron {
    weights: Vec<Value>,
    bias: Value,
    output: Value,
}

fn new_neuron<R: SeedableRng + RngCore>(inputs: &[&Value], activation: fn(&Value) -> Value, rng: &mut R) -> Neuron {
    let weights: Vec<_> = inputs.iter().map(|_| Value::new(rng.gen_range(-1.0..1.0))).collect();
    let bias = Value::new(rng.gen_range(-1.0..1.0));
    let result_value = weights.iter().zip(inputs.iter()).fold(bias.clone(), |ref acc, (w, i)| acc + &(w * i));
    let output = activation(&result_value);
    Neuron { weights: weights.clone(), bias: bias.clone(), output }
}


#[cfg(test)]
mod tests {
    use std::iter::once;
    use rand::distributions::{Distribution, Standard};
    use rand::{Rng, SeedableRng};
    use rand::prelude::StdRng;
    use crate::{backprop, reset, Value};
    use crate::value_types::input::InputValue;
    use crate::value_types::tanh::tanh;
    use super::{Neuron, new_neuron};

    fn gen_lin_test_data() -> Vec<(f32, f32)> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(1234);
        let sample_count = 20000;
        let mut result = Vec::with_capacity(sample_count);
        for _ in 0..sample_count {
            let x = rng.gen_range(0.0..1.0);
            let y = x*2.0 + rng.gen_range(-1.0..1.0) + 12.0;
            let value = (x, y);
            result.push(value);
        }
        result
    }


    fn simple_scatter_plot(show: bool) {
        let n: usize = 100;
        let t: Vec<f64> = linspace(0., 10., n).collect();
        let y = t.iter().map(|x| x.sin()).collect();

        let trace = Scatter::new(t, y).mode(Mode::Markers);
        let mut plot = Plot::new();
        plot.add_trace(trace);
        plot.show();
        // println!("{}", plot.to_inline_html(Some("simple_scatter_plot")));
    }


    fn gen_test_data(sample_count: usize) -> Vec<(f32, f32, bool)> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(1234);
        let dist = rand::distributions::Uniform::new(-1.0, 1.0);
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
    fn test_neuron_line() {
        let x = Value::new(1.0);
        let w = Value::new(4.0);
        let b = Value::new(2.0);
        let y = &(&x * &w) + &b;

        let mut test_data = gen_lin_test_data();
        let mut epoch = 0;
        let batch_size = 20;
        let step_size = 0.1;
        let epoch_count = 4;

        let expected = Value::new(1.0);

        let mut loss = &y - &expected;
        loss = &loss * &loss;

        for _ in 0..epoch_count {
            let mut error = 0.0;
            epoch += 1;
            for batch in test_data.chunks(batch_size) {
                let mut grad_w = 0.0;
                let mut grad_b = 0.0;
                for (i, (xv, yv)) in batch.iter().enumerate() {
                    x.set_value(*xv);
                    expected.set_value(*yv);
                    loss.forward();
                    error += loss.value();
                    backprop(&loss);
                    grad_w += w.grad().get() / batch_size as f32;
                    grad_b += b.grad().get() / batch_size as f32;
                    reset(&loss);
                }
                w.set_value(w.value() - step_size * grad_w);
                b.set_value(b.value() - step_size * grad_b);
                grad_w = 0.0;
                grad_b = 0.0;
            }
            println!("epoch: {}, error: {}, w: {}, b: {}", epoch, error / test_data.len() as f32, w.value(), b.value());
        }
    }
    #[test]
    fn test_neuron_shape() {
        let x = Value::new(1.0);
        let y = Value::new(-2.0);
        let inputs = vec![&x, &y];

        let mut rng = StdRng::seed_from_u64(42);
        let mut hidden_layer: Vec<Neuron> = Vec::default();
        for _ in 0..3 {
            hidden_layer.push(new_neuron(&inputs, tanh, &mut rng));
        }

        let outputs = hidden_layer.iter().map(|n| &n.output).collect::<Vec<_>>();
        let output = new_neuron(&outputs, tanh, &mut rng);

        let training_data = gen_test_data(5000);
        let test_data = gen_test_data(5000);

        let mut epoch = 0;
        let batch_size = 10;
        let step_size = 0.03;

        let mut weights = Vec::default();
        for n in hidden_layer.iter() {
            weights.extend(n.weights.iter());
            weights.push(&n.bias);
        }
        weights.extend(output.weights.iter());
        weights.push(&output.bias);

        let expected = Value::new(1.0);

        let mut loss = &output.output - &expected;
        loss = &loss * &loss;
        let epoch_count = 200;

        for _ in 0..epoch_count {
            let mut error = 0.0;
            epoch += 1;
            for batch in training_data.chunks(batch_size) {
                let mut grads = vec![0.0; weights.len()];
                for (i, (xv, yv, expectedv)) in batch.iter().enumerate() {
                    x.set_value(*xv);
                    y.set_value(*yv);
                    let expectedv = (if *expectedv { 1.0 } else { -1.0 });
                    expected.set_value(expectedv);

                    loss.forward();
                    error += loss.value();
                    backprop(&loss);
                    for (j, w) in weights.iter().enumerate() {
                        grads[j] += w.grad().get() / batch_size as f32;
                    }
                    reset(&loss);
                }
                for (j, w) in weights.iter().enumerate() {
                    w.set_value(w.value() - step_size * grads[j]);
                }
                grads.fill(0.0);
            }
            println!("epoch: {}, error: {}", epoch, error / training_data.len() as f32);
        }
        // for (xv, yv, exp) in &test_data[0..1000] {
        //     x.set_value(*xv);
        //     y.set_value(*yv);
        //     expected.set_value(if *exp { 1.0 } else { -1.0 });
        //     output.output.forward();
        //     println!("{},{},{}", xv, yv, output.output.value());
        // }
    }
}