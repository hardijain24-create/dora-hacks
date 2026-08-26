const fs = require('fs');
const tf = require('@tensorflow/tfjs');

async function run() {
    // 1. Load labels.json
    const labels = JSON.parse(fs.readFileSync('public/model/labels.json', 'utf8'));

    // 2. Load model
    // Ensure L2 regularizer compatibility (as in lib/isl/model.ts)
    const l1l2Instance = tf.regularizers.l1l2({ l1: 0, l2: 1e-4 });
    const L1L2Ctor = l1l2Instance.constructor;
    const L2Regularizer = (function(Base) {
      class L2Class extends Base {
        constructor(config) { super(config); }
        getClassName() { return 'L2'; }
        static get className() { return 'L2'; }
        static fromConfig(cls, config) { return new cls({ l1: 0, l2: config.l2 || config.l || 0 }); }
      }
      return L2Class;
    })(L1L2Ctor);
    tf.serialization.registerClass(L2Regularizer);
    const modelUrl = 'http://localhost:8000/public/model/model.json';
    const model = await tf.loadLayersModel(modelUrl);
    
    const seqs = JSON.parse(fs.readFileSync('scratch/test_sequences.json', 'utf8'));
    
    let correct = 0;
    let total = 0;

    console.log("| Expected INCLUDE | Predicted | Confidence | Correct |");
    console.log("|-------------------|-----------|------------|---------|");

    let seqEntries = [];
    if (Array.isArray(seqs)) {
        // Single sequence
        seqEntries = [["50. Yellow/MVI_3728.MOV", seqs]];
    } else {
        seqEntries = Object.entries(seqs);
    }

    for (const [key, seqData] of seqEntries) {
        const expectedLabel = key.split('/')[0]; // Folder name is label

        // shape is [40, 258]
        const inputTensor = tf.tensor([seqData]); // [1, 40, 258]
        
        const raw = model.predict(inputTensor);
        const output = Array.isArray(raw) ? raw[0] : raw;
        const probs = output.dataSync();
        
        let maxIdx = 0;
        let maxVal = probs[0];
        for (let i = 1; i < probs.length; i++) {
            if (probs[i] > maxVal) {
                maxVal = probs[i];
                maxIdx = i;
            }
        }
        
        const predictedLabel = labels[maxIdx];
        const isCorrect = (expectedLabel === predictedLabel) ? "YES" : "NO";
        
        if (isCorrect === "YES") correct++;
        total++;

        console.log(`| ${expectedLabel.padEnd(17)} | ${predictedLabel.padEnd(9)} | ${(maxVal).toFixed(2).padEnd(10)} | ${isCorrect.padEnd(7)} |`);
        
        tf.dispose(inputTensor);
        tf.dispose(output);
    }

    console.log(`\nAccuracy: ${correct}/${total} = ${Math.round((correct/total)*100)}%`);
}

run().catch(console.error);
