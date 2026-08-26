import fs from 'fs';
import path from 'path';
import * as tf from '@tensorflow/tfjs-node';

const DATA_DIR = 'data/mvp_features';
const OUT_DIR = 'public/models/isl-mvp';

// Simple .npy parser
function parseNpy(buffer, dataType) {
  // NPY files start with \x93NUMPY\x01\x00
  // Then a 2-byte header length (little endian)
  const headerLen = buffer.readUInt16LE(8);
  const headerStr = buffer.toString('ascii', 10, 10 + headerLen);
  
  const dataOffset = 10 + headerLen;
  
  if (dataType === 'float32') {
    return new Float32Array(buffer.buffer, buffer.byteOffset + dataOffset, (buffer.byteLength - dataOffset) / 4);
  } else if (dataType === 'int32') {
    return new Int32Array(buffer.buffer, buffer.byteOffset + dataOffset, (buffer.byteLength - dataOffset) / 4);
  }
}

async function main() {
  console.log('Loading dataset...');
  
  const reportPath = path.join(DATA_DIR, 'report.json');
  if (!fs.existsSync(reportPath)) {
    console.error('report.json not found. Run 02_extract_features.py first.');
    process.exit(1);
  }
  
  const report = JSON.parse(fs.readFileSync(reportPath, 'utf8'));
  const numSamples = report.total_samples;
  const numClasses = report.classes.length;
  
  console.log(`Samples: ${numSamples}, Classes: ${numClasses}`);
  
  const xBuffer = fs.readFileSync(path.join(DATA_DIR, 'X.npy'));
  const yBuffer = fs.readFileSync(path.join(DATA_DIR, 'y.npy'));
  
  const xData = parseNpy(xBuffer, 'float32');
  const yData = parseNpy(yBuffer, 'int32');
  
  const xs = tf.tensor3d(xData, [numSamples, 40, 258]);
  const ys = tf.tensor1d(yData, 'int32');
  
  console.log('Building model...');
  
  const model = tf.sequential();
  
  model.add(tf.layers.conv1d({
    inputShape: [40, 258],
    filters: 64,
    kernelSize: 3,
    activation: 'relu'
  }));
  model.add(tf.layers.maxPooling1d({poolSize: 2}));
  model.add(tf.layers.lstm({units: 64, returnSequences: false}));
  model.add(tf.layers.dropout({rate: 0.2}));
  model.add(tf.layers.dense({units: 32, activation: 'relu'}));
  model.add(tf.layers.dense({units: numClasses, activation: 'softmax'}));
  
  model.compile({
    optimizer: tf.train.adam(0.001),
    loss: 'sparseCategoricalCrossentropy',
    metrics: ['accuracy']
  });
  
  model.summary();
  
  console.log('Training model...');
  
  await model.fit(xs, ys, {
    epochs: 100,
    batchSize: 32,
    validationSplit: 0.2,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        if (epoch % 10 === 0 || epoch === 99) {
          console.log(`Epoch ${epoch + 1}/100 - loss: ${logs.loss.toFixed(4)} - acc: ${logs.acc.toFixed(4)} - val_loss: ${(logs.val_loss||0).toFixed(4)} - val_acc: ${(logs.val_acc||0).toFixed(4)}`);
        }
      }
    }
  });
  
  console.log('\\nTraining complete. Exporting model...');
  
  if (!fs.existsSync(OUT_DIR)) {
    fs.mkdirSync(OUT_DIR, { recursive: true });
  }
  
  await model.save(`file://${OUT_DIR}`);
  
  // Copy labels over
  fs.copyFileSync(path.join(DATA_DIR, 'labels.json'), path.join(OUT_DIR, 'labels.json'));
  
  console.log(`\\nModel exported to ${OUT_DIR}/`);
  console.log('MVP pipeline is now ready. The UI can safely switch to this model once verified.');
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
