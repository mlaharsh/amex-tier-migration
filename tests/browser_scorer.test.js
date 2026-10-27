/* Node parity checks for the static browser scorer.
 *
 * Run with: node tests/browser_scorer.test.js
 */
"use strict";

const fs = require("fs");
const path = require("path");
const vm = require("vm");
const assert = require("assert");

const root = path.resolve(__dirname, "..");
const scorer = require(path.join(root, "docs", "scorer.js"));
const sandbox = { window: {} };
vm.createContext(sandbox);
vm.runInContext(
  fs.readFileSync(path.join(root, "docs", "scorer_data.js"), "utf8"),
  sandbox
);
const artifact = sandbox.window.CARD_MODEL;

assert.ok(artifact, "scorer data must be present");
assert.strictEqual(artifact.numeric.length + artifact.categorical.length, 28);

for (const validation of artifact.validation) {
  const result = scorer.scoreFeatures(validation.features, artifact);
  assert.ok(
    Math.abs(result.rawProbability - validation.expectedProbability) < 1e-12,
    `${validation.id}: expected ${validation.expectedProbability}, got ${result.rawProbability}`
  );
  assert.ok(
    result.rawProbability >= 0 && result.rawProbability <= 1,
    `${validation.id}: probability must be bounded`
  );
}

for (const example of artifact.examples) {
  const result = scorer.scoreRaw(example.raw, artifact);
  const validation = artifact.validation.find((item) => item.id === example.id);
  assert.ok(validation, `${example.id}: validation case is missing`);
  assert.ok(
    Math.abs(result.rawProbability - validation.expectedProbability) < 1e-12,
    `${example.id}: raw example drifted from the exported validation case`
  );
}

const selected = scorer.thresholdMetrics(artifact, artifact.metadata.threshold);
assert.ok(selected.contacts > 0, "threshold lab should contact some customers");
assert.ok(selected.precision >= 0 && selected.precision <= 1);
assert.ok(selected.recall >= 0 && selected.recall <= 1);

console.log("Browser scorer parity checks passed.");
