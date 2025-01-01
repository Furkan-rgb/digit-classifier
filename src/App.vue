<template>
  <div class="container">
    <h2>MNIST Classifier using a CNN model.</h2>
    <p>Draw a digit.</p>

    <!-- Main layout: left side (canvas/buttons) + right side (layers + digit gauges) -->
    <div class="main-layout">
      <!-- Left Column: Canvas + Clear Button + Prediction Text -->
      <div class="canvas-column">
        <canvas
          ref="drawingCanvas"
          :width="canvasWidth"
          :height="canvasHeight"
          @mousedown="startDrawing"
          @mousemove="draw"
          @mouseup="stopDrawing"
          style="border: 2px solid #444; cursor: crosshair"
        ></canvas>

        <div class="buttons">
          <button @click="clearCanvas">Clear</button>
        </div>

        <div class="prediction" v-if="prediction !== null">
          <h2>Predicted Digit: {{ prediction }}</h2>
        </div>
      </div>

      <!-- Right Column: CNN Layers (vertical) + Probability Gauges -->
      <div class="pipeline-column">
        <!-- CNN Overview (vertical stacking, smaller spacing) -->
        <div class="cnn-overview vertical">
          <!-- 1) INPUT NODE -->
          <div class="layer input-layer">
            <div class="single-node">Input</div>
          </div>

          <!-- Lines: Input -> Conv16 (16 filters) -->
          <svg class="connections">
            <line
              v-for="n in 16"
              :key="n"
              x1="50%"
              y1="0"
              :x2="((n - 0.5) / 16) * 100 + '%'"
              y2="100%"
              stroke="#999"
              stroke-width="1"
            />
          </svg>

          <!-- 2) CONV 16 FILTERS (16 dots) -->
          <div class="layer conv16-layer">
            <div
              v-for="n in 16"
              :key="n"
              class="filter-dot"
              :title="'Filter ' + n"
            ></div>
          </div>

          <!-- Lines: Conv16 (16 filters) -> Pool1 (single node) -->
          <svg class="connections">
            <line
              v-for="n in 16"
              :key="n"
              :x1="((n - 0.5) / 16) * 100 + '%'"
              y1="0"
              x2="50%"
              y2="100%"
              stroke="#aaa"
              stroke-width="1"
            />
          </svg>

          <!-- 3) MAXPOOL(2×2) LAYER (single node) -->
          <div class="layer pool1-layer">
            <div class="single-node">MaxPool(2×2)</div>
          </div>

          <!-- Lines: Pool1 (single node) -> Conv32 (32 filters) -->
          <svg class="connections">
            <line
              v-for="n in 32"
              :key="n"
              x1="50%"
              y1="0"
              :x2="((n - 0.5) / 32) * 100 + '%'"
              y2="100%"
              stroke="#999"
              stroke-width="1"
            />
          </svg>

          <!-- 4) CONV 32 FILTERS (32 dots) -->
          <div class="layer conv32-layer">
            <div
              v-for="n in 32"
              :key="n"
              class="filter-dot"
              :title="'Filter ' + n"
            ></div>
          </div>

          <!-- Lines: Conv32 (32 filters) -> Pool2 (single node) -->
          <svg class="connections">
            <line
              v-for="n in 32"
              :key="n"
              :x1="((n - 0.5) / 32) * 100 + '%'"
              y1="0"
              x2="50%"
              y2="100%"
              stroke="#bbb"
              stroke-width="1"
            />
          </svg>

          <!-- 5) MAXPOOL(2×2) LAYER (single node) -->
          <div class="layer pool2-layer">
            <div class="single-node">MaxPool(2×2)</div>
          </div>

          <!-- Lines: Pool2 -> each digit gauge (10 lines) -->
          <svg class="connections to-digit-gauges">
            <line
              v-for="(val, idx) in probabilities"
              :key="idx"
              x1="50%"
              y1="0"
              :x2="((idx + 0.5) / probabilities.length) * 100 + '%'"
              y2="100%"
              stroke="#666"
              stroke-width="1"
            />
          </svg>
        </div>

        <!-- Probability Gauges for digits 0-9 -->
        <div class="digit-gauges">
          <div
            class="digit-gauge"
            v-for="(val, idx) in probabilities"
            :key="idx"
          >
            <div class="digit-bar-outer">
              <div
                class="digit-bar-fill"
                :style="{ height: (val * 100).toFixed(1) + '%' }"
              ></div>
            </div>
            <div class="digit-label">Digit {{ idx }}</div>
          </div>
        </div>
      </div>
    </div>

    <!-- Activation Maps below (unchanged logic) -->
    <!--
    <div class="activations" v-if="activationMaps.length > 0">
      ...
    </div>
    -->
  </div>

  <section class="explanations">
    <h2>How This Model Was Trained</h2>
    <p>
      This MNIST digit classifier uses a custom CNN architecture built with
      <strong>TensorFlow Keras</strong>.
    </p>
    <ul>
      <li>
        <strong>Convolution + MaxPooling:</strong> We use two convolutional
        blocks (16 and 32 filters) to extract features, each followed by a 2×2
        MaxPooling layer to reduce spatial dimensions.
      </li>
      <li>
        <strong>Flatten + Dense Layer:</strong> We flatten the output of the
        last convolution block and feed it into a <code>Dense</code> layer with
        10 units for final classification.
      </li>
      <li>
        <strong>Data Augmentation:</strong> On the training set, we apply random
        rotation, translation, zoom, and contrast adjustments to increase the
        variety of inputs and improve generalization.
      </li>
    </ul>
    <p>
      Below is the <strong>exact Python code</strong> used to build and train
      this model. After training, we converted it to the
      <code>TensorFlow.js</code> format so it can run directly in your browser:
    </p>
    <div class="code-container">
      <pre><code>
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import tensorflow as tf
from tensorflow.keras import layers, Model
import tensorflowjs as tfjs
import pathlib

def build_keras_model():
    # Input shape: [B, 28, 28, 1]
    inputs = layers.Input(shape=(28, 28, 1))

    c1 = layers.Conv2D(filters=16, kernel_size=3, activation='relu')(inputs)
    # => shape [B, 26, 26, 16]
    c1 = layers.MaxPooling2D(pool_size=2, name="c1_output")(c1)  

    c2 = layers.Conv2D(filters=32, kernel_size=3, activation='relu')(c1)
    # => shape [B, 24, 24, 32]
    c2 = layers.MaxPooling2D(pool_size=2, name="c2_output")(c2)
    
    # Flatten
    x = layers.Flatten()(c2)
    logits = layers.Dense(10, name="output")(x)  # => shape [B, 10]

    # Return all three: logits, c1, c2
    model = Model(inputs=inputs, outputs=[logits, c1, c2])
    return model

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(factor=0.05),
    tf.keras.layers.RandomTranslation(0.1, 0.1),
    tf.keras.layers.RandomZoom(height_factor=(-0.1, 0.1)),
    tf.keras.layers.RandomContrast(0.1),
])

def augment(image, label):
    image = data_augmentation(image)
    return image, label

def get_mnist_datasets(batch_size=64):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train[..., None] / 255.0
    x_test = x_test[..., None] / 255.0

    train_ds = (tf.data.Dataset.from_tensor_slices((x_train, y_train))
                .map(augment)
                .shuffle(60000)
                .batch(batch_size)
                .prefetch(tf.data.AUTOTUNE))
    test_ds = (tf.data.Dataset.from_tensor_slices((x_test, y_test))
               .batch(batch_size)
               .prefetch(tf.data.AUTOTUNE))

    return train_ds, test_ds

def train_keras_model(epochs=2, batch_size=64):
    train_ds, test_ds = get_mnist_datasets(batch_size)
    model = build_keras_model()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=[
            tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            None,  # No loss for c1
            None   # No loss for c2
        ],
        metrics={'output': 'accuracy'}
    )

    # Train
    model.fit(train_ds, epochs=epochs)

    # Evaluate
    test_loss, test_logits_loss, test_accuracy = model.evaluate(test_ds)
    print(f"\\nTest accuracy: {test_accuracy:.2%}")

    return model

if __name__ == "__main__":
    trained_model = train_keras_model(epochs=10)
    tfjs.converters.save_keras_model(trained_model, "tfjs_model")
    print("Saved TensorFlow.js model to frontend/public/tfjs_model")
      </code></pre>
    </div>
    <p>
      Finally, we load this model (<code>model.json</code>) on the front end and
      run inference on the drawing you create in the canvas above. This
      demonstrates the power of <strong>TensorFlow.js</strong> to do deep
      learning inference <em>in-browser</em>.
    </p>
  </section>
</template>

<script setup>
import { ref, onMounted, nextTick } from "vue";
import * as tf from "@tensorflow/tfjs";

const canvasWidth = 280;
const canvasHeight = 280;
const mapWidth = 52;
const mapHeight = 52;

const drawingCanvas = ref(null);
let ctx = null;
let drawing = false;

let model = null;

const probabilities = ref(new Array(10).fill(0));
const prediction = ref(null);
const activationMaps = ref([]);

// On mount: set up the canvas and load the TF.js model
onMounted(async () => {
  if (drawingCanvas.value) {
    ctx = drawingCanvas.value.getContext("2d");
    // Fill with white
    ctx.fillStyle = "#fff";
    ctx.fillRect(0, 0, canvasWidth, canvasHeight);
  }

  try {
    model = await tf.loadLayersModel("tfjs_model/model.json");
    console.log("Model loaded:", model);
  } catch (err) {
    console.error("Error loading model:", err);
  }
});

function startDrawing(e) {
  if (!ctx) return;
  drawing = true;
  ctx.beginPath();
  ctx.moveTo(e.offsetX, e.offsetY);
  runInference();
}

function draw(e) {
  if (!drawing || !ctx) return;
  ctx.lineWidth = 12;
  ctx.lineCap = "round";
  ctx.strokeStyle = "#000";
  ctx.lineTo(e.offsetX, e.offsetY);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(e.offsetX, e.offsetY);

  runInference();
}

function stopDrawing() {
  drawing = false;
  runInference();
}

function clearCanvas() {
  if (!ctx) return;
  ctx.fillStyle = "#fff";
  ctx.fillRect(0, 0, canvasWidth, canvasHeight);

  probabilities.value = new Array(10).fill(0);
  prediction.value = null;
  activationMaps.value = [];
}

// Run inference
async function runInference() {
  if (!model || !drawingCanvas.value) {
    return;
  }

  const smallCanvas = document.createElement("canvas");
  smallCanvas.width = 28;
  smallCanvas.height = 28;
  const smallCtx = smallCanvas.getContext("2d");
  smallCtx.drawImage(drawingCanvas.value, 0, 0, 28, 28);

  const imgData = smallCtx.getImageData(0, 0, 28, 28);
  const inputBuffer = new Float32Array(28 * 28);
  for (let i = 0; i < 28 * 28; i++) {
    const idx = i * 4;
    const r = imgData.data[idx];
    inputBuffer[i] = (255 - r) / 255.0;
  }
  const inputTensor = tf.tensor4d(inputBuffer, [1, 28, 28, 1]);

  let results;
  try {
    results = model.predict(inputTensor);
  } catch (err) {
    console.error("Predict error:", err);
    tf.dispose(inputTensor);
    return;
  }

  // Check for multiple outputs
  let logits, c1, c2;
  if (Array.isArray(results)) {
    [logits, c1, c2] = results;
  } else {
    logits = results["Identity"] || results["output_0"] || null;
    c1 = results["Identity_1"] || results["output_1"] || null;
    c2 = results["Identity_2"] || results["output_2"] || null;
  }

  if (!logits) {
    console.warn("No logits found.");
    tf.dispose([inputTensor, results]);
    return;
  }

  // Manual softmax
  const logitsData = await logits.data();
  const exps = logitsData.map((x) => Math.exp(x));
  const sumExps = exps.reduce((a, b) => a + b, 0);
  const probs = exps.map((x) => x / sumExps);
  probabilities.value = probs;

  // Argmax => predicted digit
  prediction.value = probs.indexOf(Math.max(...probs));

  // Activation maps
  activationMaps.value = [];
  if (c1) {
    const c1Data = await c1.data();
    activationMaps.value.push(extractConv2DActivationMaps(c1Data, c1.shape));
  }
  if (c2) {
    const c2Data = await c2.data();
    activationMaps.value.push(extractConv2DActivationMaps(c2Data, c2.shape));
  }

  tf.dispose([inputTensor, results, logits, c1, c2]);

  await nextTick();
  drawActivationMapsOnCanvas();
}

function extractConv2DActivationMaps(floatData, shape) {
  // shape [1, H, W, C]
  const [_, H, W, C] = shape;
  const out = [];
  let idx = 0;
  for (let channel = 0; channel < C; channel++) {
    const channelData = new Array(H * W);
    for (let i = 0; i < H * W; i++) {
      channelData[i] = floatData[idx++];
    }
    out.push(channelData);
  }
  return out;
}

function drawActivationMapsOnCanvas() {
  const layerEls = document.querySelectorAll(".layer-container");

  layerEls.forEach((layerEl, layerIndex) => {
    const mapGrid = layerEl.querySelector(".map-grid");
    if (!mapGrid) return;
    const layerData = activationMaps.value[layerIndex];
    const channelCanvases = mapGrid.querySelectorAll("canvas");

    channelCanvases.forEach((cv, chanIndex) => {
      const c = cv.getContext("2d");
      if (!c) return;
      const channelData = layerData[chanIndex];
      if (!channelData) return;

      const length = channelData.length;
      const H = Math.sqrt(length);
      const W = H;
      const minVal = Math.min(...channelData);
      const maxVal = Math.max(...channelData);
      const range = maxVal - minVal || 1e-5;

      const buf = new Uint8ClampedArray(length * 4);
      for (let i = 0; i < length; i++) {
        const val = (channelData[i] - minVal) / range; // 0..1
        const gray = Math.round(val * 255);
        buf[i * 4 + 0] = gray;
        buf[i * 4 + 1] = gray;
        buf[i * 4 + 2] = gray;
        buf[i * 4 + 3] = 255; // A
      }
      const imgData = new ImageData(buf, W, H);

      const tmpCanvas = document.createElement("canvas");
      tmpCanvas.width = W;
      tmpCanvas.height = H;
      const tmpCtx = tmpCanvas.getContext("2d");
      tmpCtx.putImageData(imgData, 0, 0);

      c.clearRect(0, 0, mapWidth, mapHeight);
      c.drawImage(tmpCanvas, 0, 0, mapWidth, mapHeight);
    });
  });
}
</script>

<style scoped>
/* Overall container */
.container {
  width: 100%;
  max-width: 1200px;
  margin: 2rem auto;
  text-align: center;
  font-family: sans-serif;
}

/* Main layout */
.main-layout {
  /* Removed 'flex-wrap: wrap;' so columns stay side by side */
  display: flex;
  gap: 2rem;
  align-items: flex-start; /* Ensures top alignment if columns differ in height */
}

.canvas-column {
  flex: 0 0 auto;
  min-width: 280px;
  text-align: center;
}

.buttons {
  margin-top: 1rem;
}

.prediction {
  margin-top: 1rem;
  font-weight: bold;
}

.pipeline-column {
  flex: 0 0 auto;
  min-width: 300px;
}

/* CNN Overview (vertical, smaller spacing) */
.cnn-overview.vertical {
  display: flex;
  flex-direction: column;
  align-items: center;
  position: relative;
}

/* Each “layer” block */
.layer {
  margin: 0.2rem 0; /* smaller vertical margin */
  display: flex;
  justify-content: center;
  flex-wrap: wrap;
}

/* Single node (Input, Pool, etc.) */
.single-node {
  padding: 4px 8px;
  border: 1px solid #999;
  border-radius: 6px;
  background-color: #f5f5f5;
  font-size: 0.85rem;
  font-weight: bold;
}

/* Represent filters as small circular dots */
.filter-dot {
  width: 6px;
  height: 6px;
  background: #666;
  border-radius: 50%;
  margin: 2px;
}

/* Vertical connections between layers, smaller height, thinner line */
.connections {
  width: 120px;
  height: 30px; /* smaller gap to reduce total height */
  overflow: visible;
}

/* Probability Gauges */
.digit-gauges {
  margin: 2rem 0 1rem;
  display: flex;
  gap: 12px;
  justify-content: center;
  align-items: flex-end;
}

.digit-gauge {
  width: 30px;
  display: flex;
  flex-direction: column;
  align-items: center;
  font-size: 0.85rem;
}

.digit-bar-outer {
  position: relative;
  width: 100%;
  height: 80px; /* smaller if you want less vertical space */
  background: #eee;
  border: 1px solid #ccc;
  border-radius: 4px;
  overflow: hidden;
}

.digit-bar-fill {
  position: absolute;
  left: 0;
  bottom: 0;
  width: 100%;
  background-color: #2196f3;
  transition: height 0.3s ease;
}

.digit-label {
  margin-top: 4px;
}

/* Activation Maps */
.activations {
  margin-top: 2rem;
  text-align: left;
}

.layer-container {
  margin-bottom: 1rem;
}

.map-grid {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
  margin: 8px 0;
}

.map-grid canvas {
  background: #ccc;
  display: block;
  border: 1px solid #999;
}
.explanations {
  margin-top: 2rem;
  text-align: left;
  background: #fff;
  padding: 1rem;
  border-radius: 8px;
  box-shadow: 0px 2px 6px rgba(0, 0, 0, 0.1);
}

/* A container for your code snippet; tweak to your liking. */
.code-container {
  background-color: #f5f5f5;
  padding: 1rem;
  margin-top: 1rem;
  border-radius: 6px;
  overflow-x: auto; /* Horizontal scrolling for wide code. */
}

.code-container pre {
  margin: 0;
  font-family: "Source Code Pro", monospace;
  font-size: 0.9rem;
  line-height: 1.4em;
}
</style>
