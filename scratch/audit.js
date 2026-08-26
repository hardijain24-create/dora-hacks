const fs = require('fs');

async function runAudit() {
  const labels = JSON.parse(fs.readFileSync('public/model/labels.json', 'utf8'));
  console.log(`LABELS.JSON CLASSES: ${labels.length}`);

  const modelJson = JSON.parse(fs.readFileSync('public/model/model.json', 'utf8'));
  // Find the last Dense layer to see output shape
  const layers = modelJson.topology.keras_version ? modelJson.topology.config.layers : [];
  let outputDim = null;
  for (let i = layers.length - 1; i >= 0; i--) {
    if (layers[i].className === 'Dense') {
      outputDim = layers[i].config.units;
      break;
    }
  }
  console.log(`MODEL OUTPUT CLASSES (units in last dense layer): ${outputDim}`);

  if (labels.length !== outputDim) {
    console.error(`STOP: Model output dimension ${outputDim} != labels.json count ${labels.length}`);
  } else {
    console.log(`PASS: Model output dimension matches labels.json count (${labels.length}).`);
  }
}

runAudit();
