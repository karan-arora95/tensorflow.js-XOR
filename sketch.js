let model = tf.sequential();
let inputs = [];

const train_xs = tf.tensor2d([[0, 0], [0, 1], [1, 0], [1, 1]]);
const train_ys = tf.tensor2d([[0], [1], [1], [0]]);

let resolution = 30, cols, rows;
let iterations = 1000;

function setup() {

  createCanvas(400, 400);

  let hidden = tf.layers.dense({
    inputShape: [2],
    units: 4,
    activation: 'sigmoid'
  });
  
  let output = tf.layers.dense({
    units: 1,
    activation: 'sigmoid'
  });
  
  model.add(hidden);
  model.add(output);
  
  model.compile({
    optimizer: tf.train.adam(0.1),
    loss: 'meanSquaredError'
  });

  cols = width / resolution;
  rows = height / resolution;
  for(let i = 0; i < cols; i++) {
    for(let j = 0; j < rows; j++) {
      let x1 = i / cols;
      let x2 = j / rows;
      inputs.push([x1, x2]);
    }
  }

}

function trainModel() {
  return model.fit(train_xs, train_ys, {
    shuffle: true,
    epocs: 2
  });
}

function draw() {
  background(0);

  // Train The Model
  trainModel().then((result) => {
    // Print the loss
    console.log(result.history.loss[0]);
    iterations--;
    if(iterations > 0) {
      draw();
    }
  });

  // Clean up not needed tensors :)
  tf.tidy(() => {

    let ys = model.predict(tf.tensor2d(inputs)).dataSync();

    let index = 0;
    for(let i = 0; i < cols; i++) {
      for(let j = 0; j < rows; j++) {
        let br = ys[index] * 255;
        fill(br);
        rect(i * resolution, j * resolution, resolution, resolution);
        fill(255 - br);
        textSize(8);  
        textAlign(CENTER, CENTER);
        text(nf(ys[index], 1, 2), i * resolution + resolution/2, j * resolution + resolution/2);
        index++;
      }
    }

  });
  noLoop();

}