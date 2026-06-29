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
          class="drawing-canvas"
          :width="canvasWidth"
          :height="canvasHeight"
          @mousedown="startDrawing"
          @mousemove="draw"
          @mouseup="stopDrawing"
          @mouseleave="stopDrawing"
          @touchstart.prevent="startDrawing"
          @touchmove.prevent="draw"
          @touchend.prevent="stopDrawing"
        ></canvas>

        <div class="buttons">
          <button @click="clearCanvas">Clear</button>
        </div>

        <div class="prediction" v-if="prediction !== null">
          <h2>Predicted Digit: {{ prediction }}</h2>
        </div>
      </div>

      <!-- Right: live confidence for digits 0-9 -->
      <div class="results-column">
        <span class="panel-caption">Confidence</span>
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
            <div class="digit-label"><span class="digit-word">Digit </span>{{ idx }}</div>
          </div>
        </div>
      </div>
    </div>

    <!-- A real window into the CNN: the low-res input it actually receives,
         plus the live convolutional feature maps for the current drawing -->
    <div class="network-view">
      <h3 class="network-view-title">What the network sees</h3>

      <div class="net-section net-input">
        <span class="panel-caption">Model input (28×28)</span>
        <canvas
          ref="inputPreview"
          class="input-preview"
          width="28"
          height="28"
        ></canvas>
      </div>

      <div class="net-section layer-container">
        <span class="panel-caption">Feature maps · 16 filters</span>
        <div class="map-grid">
          <canvas
            v-for="n in 16"
            :key="'l1-' + n"
            :width="mapWidth"
            :height="mapHeight"
          ></canvas>
        </div>
      </div>

      <p class="net-hint">
        Each tile shows how one of the 16 first-layer filters responds to your
        digit — most latch onto a particular edge or stroke direction. Colors
        run cool to warm: dark purple is no response, bright orange and yellow
        are strong activations. Draw above to watch them light up.
      </p>
    </div>
  </div>

  <section class="explanations">
    <h2>How This Model Was Trained</h2>
    <p>
      This MNIST digit classifier uses a custom CNN architecture built with
      <strong>TensorFlow Keras</strong>.
    </p>
    <ul>
      <li>
        <strong>Convolution + MaxPooling:</strong> I've used two convolutional
        blocks (16 and 32 filters) to extract features, each followed by a 2×2
        MaxPooling layer to reduce spatial dimensions.
      </li>
      <li>
        <strong>Flatten + Dense Layer:</strong> Then I flatten the output of the
        last convolution block and feed it into a <code>Dense</code> layer with
        10 units for final classification.
      </li>
      <li>
        <strong>Data Augmentation:</strong> On the training set, a random
        rotation, translation, zoom, and contrast adjustments are applied to
        increase the variety of inputs and improve generalization.
      </li>
    </ul>
    <p>
      Below is the <strong>exact Python code</strong> used to build and train
      this model. After training, the model is converted to
      <code>TensorFlow.js</code> format so it can run directly in your browser:
    </p>
    <details class="code-details">
      <summary>Show training code</summary>
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
    </details>
    <p>
      Finally, the model is loaded (<code>model.json</code>) on the front end
      and run inference on the drawing you create in the canvas.
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
const inputPreview = ref(null);
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

// Resolve a pointer position for both mouse and touch events, mapping the
// page coordinates onto the canvas' internal resolution. This keeps drawing
// accurate even when the canvas is scaled down to fit smaller screens.
function getPos(e) {
  const rect = drawingCanvas.value.getBoundingClientRect();
  const point = e.touches && e.touches.length ? e.touches[0] : e;
  const scaleX = drawingCanvas.value.width / rect.width;
  const scaleY = drawingCanvas.value.height / rect.height;
  return {
    x: (point.clientX - rect.left) * scaleX,
    y: (point.clientY - rect.top) * scaleY,
  };
}

function startDrawing(e) {
  if (!ctx) return;
  drawing = true;
  const { x, y } = getPos(e);
  ctx.beginPath();
  ctx.moveTo(x, y);
  runInference();
}

function draw(e) {
  if (!drawing || !ctx) return;
  const { x, y } = getPos(e);
  ctx.lineWidth = 12;
  ctx.lineCap = "round";
  ctx.strokeStyle = "#000";
  ctx.lineTo(x, y);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(x, y);

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

  // Wipe the input preview and every feature-map thumbnail
  if (inputPreview.value) {
    const pctx = inputPreview.value.getContext("2d");
    pctx.clearRect(0, 0, inputPreview.value.width, inputPreview.value.height);
  }
  document.querySelectorAll(".map-grid canvas").forEach((cv) => {
    cv.getContext("2d").clearRect(0, 0, cv.width, cv.height);
  });
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

  // Mirror the normalized input onto the preview so visitors can see the
  // low-res, inverted image the network actually receives
  drawInputPreview(inputBuffer);

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

  // Activation maps (first conv block only; c2 is no longer displayed)
  activationMaps.value = [];
  if (c1) {
    const c1Data = await c1.data();
    activationMaps.value.push(extractConv2DActivationMaps(c1Data, c1.shape));
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

/**
 * Maps an activation value in [0, 1] to an "inferno"-style heat color
 * (dark purple -> magenta -> orange -> pale yellow). Warmer = stronger.
 */
function heatColor(t) {
  const stops = [
    [0, 0, 4],
    [87, 16, 110],
    [188, 55, 84],
    [249, 142, 9],
    [252, 255, 164],
  ];
  const x = Math.max(0, Math.min(1, t)) * (stops.length - 1);
  const i = Math.floor(x);
  const f = x - i;
  const a = stops[i];
  const b = stops[Math.min(i + 1, stops.length - 1)];
  return [
    Math.round(a[0] + (b[0] - a[0]) * f),
    Math.round(a[1] + (b[1] - a[1]) * f),
    Math.round(a[2] + (b[2] - a[2]) * f),
  ];
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
        const [r, g, b] = heatColor(val);
        buf[i * 4 + 0] = r;
        buf[i * 4 + 1] = g;
        buf[i * 4 + 2] = b;
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

/**
 * Renders the 28×28 normalized model input (white strokes on black, the way
 * the network sees it) onto the small preview canvas.
 */
function drawInputPreview(inputBuffer) {
  if (!inputPreview.value) return;
  const pctx = inputPreview.value.getContext("2d");
  const previewData = pctx.createImageData(28, 28);
  for (let i = 0; i < 28 * 28; i++) {
    const gray = Math.round(inputBuffer[i] * 255);
    previewData.data[i * 4 + 0] = gray;
    previewData.data[i * 4 + 1] = gray;
    previewData.data[i * 4 + 2] = gray;
    previewData.data[i * 4 + 3] = 255;
  }
  pctx.putImageData(previewData, 0, 0);
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

/* .main-layout just centers the two columns side-by-side */
.main-layout {
  display: flex;
  gap: 2rem;
  align-items: flex-start;
  justify-content: center; /* Center them horizontally */
}

/* Column with the canvas/drawing */
.canvas-column {
  flex: 0 0 auto;
  min-width: 280px;
  text-align: center;
}

/* Drawing surface: keep it square and let it shrink to fit the screen */
.drawing-canvas {
  border: 2px solid #444;
  cursor: crosshair;
  width: 100%;
  max-width: 280px;
  aspect-ratio: 1 / 1;
  height: auto;
  touch-action: none; /* don't scroll/zoom the page while drawing */
  user-select: none;
  -webkit-user-select: none;
  -webkit-tap-highlight-color: transparent;
}

.buttons {
  margin-top: 1rem;
}

.prediction {
  margin-top: 1rem;
  font-weight: bold;
}

/* Right-hand column holding the live confidence bars */
.results-column {
  flex: 0 0 auto;
  min-width: 300px;
}

/* Small uppercase caption above each panel/section */
.panel-caption {
  display: block;
  font-size: 0.75rem;
  font-weight: 700;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  opacity: 0.7;
  margin-bottom: 0.5rem;
}

/* Probability Gauges */
.digit-gauges {
  margin: 0rem 0 1rem;
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
  height: 80px; /* Adjust if you want less or more vertical space */
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

/* "What the network sees" panel */
.network-view {
  margin-top: 2.5rem;
  padding: 1.25rem;
  background: rgba(127, 127, 127, 0.06);
  border: 1px solid rgba(127, 127, 127, 0.18);
  border-radius: 10px;
  text-align: center;
}

.network-view-title {
  margin: 0 0 1.25rem;
}

.net-section {
  max-width: 760px;
  margin: 0 auto 1.25rem;
  text-align: center;
}

/* The low-res model input, scaled up with crisp (non-blurred) pixels */
.input-preview {
  width: 84px;
  height: 84px;
  image-rendering: pixelated;
  border: 1px solid rgba(127, 127, 127, 0.4);
  border-radius: 4px;
  background: #000;
}

/* 16 filters laid out as a tidy 4×4 grid */
.map-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 6px;
  max-width: 320px;
  margin: 0 auto;
}

.map-grid canvas {
  width: 100%;
  aspect-ratio: 1 / 1;
  height: auto;
  background: #000004; /* matches the colormap's "no response" end */
  display: block;
  border-radius: 4px;
}

.net-hint {
  max-width: 540px;
  margin: 1.25rem auto 0;
  font-size: 0.85rem;
  opacity: 0.75;
}

/* Explanations and Code Sections */
.explanations {
  margin-top: 2rem;
  text-align: left;
  background: #fff;
  padding: 1rem;
  border-radius: 8px;
  box-shadow: 0px 2px 6px rgba(0, 0, 0, 0.1);
}

/* Collapsible wrapper so the long training code stays out of the way
   (especially on phones) until a visitor chooses to expand it */
.code-details {
  margin-top: 1rem;
}

.code-details > summary {
  cursor: pointer;
  font-weight: 600;
  padding: 0.5rem 0;
  user-select: none;
  list-style: revert; /* keep the native disclosure triangle */
}

.code-container {
  background-color: #f5f5f5;
  padding: 1rem;
  margin-top: 0.5rem;
  border-radius: 6px;
  overflow-x: auto; /* Horizontal scrolling for wide code */
}

.code-container pre {
  margin: 0;
  font-family: "Source Code Pro", monospace;
  font-size: 0.9rem;
  line-height: 1.4em;
}

/* --- Mobile / narrow screens --- */
@media (max-width: 700px) {
  .container {
    margin: 1rem auto;
  }

  /* Stack the canvas and the pipeline on top of each other */
  .main-layout {
    flex-direction: column;
    align-items: center;
    gap: 1.5rem;
  }

  /* Canvas, then confidence bars, then the network panel stack naturally */
  .canvas-column,
  .results-column {
    min-width: 0;
    width: 100%;
    max-width: 360px;
  }

  .digit-gauges {
    gap: 5px;
  }

  .network-view {
    padding: 1rem 0.75rem;
  }

  .input-preview {
    width: 72px;
    height: 72px;
  }

  .map-grid {
    max-width: 280px;
  }

  .digit-gauge {
    width: auto;
    flex: 1 1 0;
    min-width: 0;
    font-size: 0.7rem;
  }

  .digit-bar-outer {
    height: 64px;
  }

  /* Drop the word "Digit" so only the number shows under each bar */
  .digit-word {
    display: none;
  }

  .explanations {
    padding: 1rem 0.85rem;
  }

  .code-container pre {
    font-size: 0.8rem;
  }
}
</style>
