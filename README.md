# MNIST Digit Classifier

An interactive MNIST digit classifier built with a custom TensorFlow convolutional neural network and deployed as a Vue application. Draw a digit with a mouse or touchscreen and watch the prediction, confidence scores, model input, and activation visualization update in real time.

**[Try the live demo →](https://furkan-rgb.github.io/digit-classifier/)**

## Highlights

- Real-time classification while you draw
- Confidence scores for all ten digit classes
- A live view of the normalized `28 × 28` image received by the model
- A visualization of first-layer convolution responses
- Mouse and touchscreen support
- Fully client-side inference—no backend or image upload required

## How it works

```text
280 × 280 drawing canvas
        ↓ resize, invert, normalize
28 × 28 × 1 input tensor
        ↓ custom CNN
10 output logits
        ↓ softmax
predicted digit + confidence scores
```

The trained model and weights are bundled with the application. TensorFlow.js loads them directly in the browser, so drawings stay on the device.

## Model architecture

| Stage | Configuration | Output shape |
| --- | --- | --- |
| Input | Grayscale image | `28 × 28 × 1` |
| Convolution | 16 filters, `3 × 3`, ReLU | `26 × 26 × 16` |
| Max pooling | `2 × 2` | `13 × 13 × 16` |
| Convolution | 32 filters, `3 × 3`, ReLU | `11 × 11 × 32` |
| Max pooling | `2 × 2` | `5 × 5 × 32` |
| Flatten | — | `800` |
| Dense | 10 linear outputs | digit logits |

The network has 12,810 trainable parameters. Its exported TensorFlow.js model exposes the classification output and intermediate activations used by the interface.

## Training recipe

The model was built with TensorFlow/Keras and trained on MNIST using:

- Pixel normalization to `[0, 1]`
- Random rotation, translation, zoom, and contrast augmentation
- Adam with a `0.001` learning rate
- Sparse categorical cross-entropy from logits
- A batch size of `64`
- `10` training epochs

The application includes the Python model and training example in its expandable “How This Model Was Trained” section. This repository contains the exported browser model rather than a standalone Python training project, dataset, or dependency manifest, and it does not publish an evaluation score.

## Tech stack

- **Vue 3** for the interface and application state
- **TensorFlow.js** for browser-based model loading and inference
- **TensorFlow/Keras** for model construction and training
- **Vite** for development and production builds
- **GitHub Pages** for static deployment

## Project structure

```text
.
├── public/
│   └── tfjs_model/
│       ├── model.json
│       └── group1-shard1of1.bin
├── src/
│   ├── App.vue          # Canvas, preprocessing, inference, and visualizations
│   ├── main.js          # Vue entry point
│   └── style.css        # Global styles
├── index.html
├── package.json
└── vite.config.js       # Vite and GitHub Pages configuration
```

## Run locally

Requirements: Node.js and npm.

```bash
git clone https://github.com/Furkan-rgb/digit-classifier.git
cd digit-classifier
npm ci
npm run dev
```

Open the local URL printed by Vite and start drawing. No API key, backend, or additional model download is required.

Create and preview a production build with:

```bash
npm run build
npm run preview
```

## Deployment

The Vite base path is configured for `/digit-classifier/`. With GitHub credentials configured, build the app and publish `dist/` to the `gh-pages` branch with:

```bash
npm run deploy
```
