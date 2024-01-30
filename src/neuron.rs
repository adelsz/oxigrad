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
    use std::vec;
    use plotters::backend::BitMapBackend;
    use plotters::chart::{ChartBuilder, LabelAreaPosition};
    use plotters::drawing::IntoDrawingArea;
    use plotters::element::Circle;
    use plotters::prelude::{BLUE, Color, IntoFont, IntoTextStyle, WHITE};
    use plotters::style::{BLACK, FontFamily, RED};
    use rand::distributions::{Distribution};
    use rand::{SeedableRng};
    use rand::prelude::StdRng;
    use crate::{backprop, reset, Value};
    use crate::value_types::tanh::tanh;
    use super::{Neuron, new_neuron};

    const SMALL_CIRCLE_RADIUS: f32 = 0.25;
    const BIG_CIRCLE_RADIUS: f32 = 0.5;

    /** Generate a dataset of points in the unit square, with a label of 1.0 if the point is in the
     * big circle, or the small circle, and -1.0 otherwise.
     */
    fn gen_data(sample_count: usize) -> Vec<(f32, f32, f32)> {
        let mut rng = StdRng::seed_from_u64(42);
        let dist = rand::distributions::Uniform::new(-1.0, 1.0);
        let mut result = Vec::with_capacity(sample_count);
        for _ in 0..sample_count {
            let x: f32 = dist.sample(&mut rng);
            let y: f32 = dist.sample(&mut rng);
            let is_big_circle = x.powi(2) + y.powi(2) < BIG_CIRCLE_RADIUS.powi(2);
            let is_small_circle = (x - 0.8).powi(2) + (y - 0.8).powi(2) < SMALL_CIRCLE_RADIUS.powi(2);
            let value = (x, y, if is_big_circle || is_small_circle { 1.0 } else { -1.0 });
            result.push(value);
        }
        result
    }

    #[test]
    fn test_neuron_shape() {
        // inputs
        let x = Value::new(1.0);
        let y = Value::new(-2.0);
        let inputs = vec![&x, &y];

        let mut rng = StdRng::seed_from_u64(42);
        // hidden layer of 3 neurons
        let hidden_layer: Vec<Neuron> = (0..3)
            .map(|_| new_neuron(&inputs, tanh, &mut rng))
            .collect();

        let outputs = hidden_layer.iter().map(|n| &n.output).collect::<Vec<_>>();
        // output neuron
        let output = new_neuron(&outputs, tanh, &mut rng);

        // generate some data and split it into training and test sets
        let mut data = gen_data(4000);
        let (training_data, test_data) = data.split_at_mut(2000);

        let batch_size = 10;
        let step_size = 0.03;

        // collect all the weights and biases into a single vector for training
        let mut weights = Vec::default();
        for n in hidden_layer.iter() {
            weights.extend(n.weights.iter());
            weights.push(&n.bias);
        }
        weights.extend(output.weights.iter());
        weights.push(&output.bias);

        // create the loss function
        let expected = Value::new(1.0);
        let mut loss = &output.output - &expected;
        loss = &loss * &loss;
        let epoch_count = 200;
        let plot_step = 2;

        // create a plot to show the training progress
        let plot = BitMapBackend::gif("test.gif", (800, 800), 200).unwrap().into_drawing_area();
        let mut ctx = ChartBuilder::on(&plot)
            .margin(40)
            .set_label_area_size(LabelAreaPosition::Left, 40)
            .set_label_area_size(LabelAreaPosition::Bottom, 40)
            .build_cartesian_2d(-1.0..1.0, -1.0..1.0)
            .unwrap();

        for epoch in 0..epoch_count {
            if epoch % plot_step == 0 {
                // plot the test data predictions each plot_step epochs. this whole block is optional and can be removed
                println!("plotting epoch {}", epoch);
                let mut test_loss = 0.0;
                let dots = test_data.iter().cloned().map(|(xv, yv, correct)| {
                    x.set_value(xv);
                    y.set_value(yv);
                    output.output.forward();
                    let result = output.output.value();
                    test_loss += (result - correct).powi(2);
                    let color = if result > 0.0 { BLUE.filled() } else { RED.filled() };
                    Circle::new((xv as f64, yv as f64), 5, color)
                });
                let font = FontFamily::SansSerif.into_font().resize(34.0);
                let big_circle_radius = ctx.backend_coord(&(BIG_CIRCLE_RADIUS as f64, 0.0)).0 - ctx.backend_coord(&(0.0, 0.0)).0;

                plot.fill(&WHITE).unwrap();
                ctx.configure_mesh().draw().unwrap();
                ctx.draw_series(dots).unwrap();
                let label = format!("Epoch: {}    Test loss: {}", epoch, test_loss/test_data.len() as f32);
                plot.draw_text(&label, &font.into_text_style(&plot), (10, 10)).unwrap();
                ctx.draw_series(once(Circle::new((0.0, 0.0), big_circle_radius, BLACK))).unwrap();
                ctx.draw_series(once(Circle::new((0.8, 0.8), big_circle_radius/2, BLACK))).unwrap();
                plot.present().unwrap();
            }
            let mut error = 0.0;
            // train the network in batches
            for batch in training_data.chunks(batch_size) {
                let mut grads = vec![0.0; weights.len()];
                for &(xv, yv, ev) in batch.iter() {
                    x.set_value(xv);
                    y.set_value(yv);
                    expected.set_value(ev);

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
    }
}