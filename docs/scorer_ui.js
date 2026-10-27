/* UI glue for the static Amex card tier migration scorer. */
(function () {
  "use strict";

  const model = window.CARD_MODEL;
  const scorer = window.CardScorer;

  const $ = (selector, root) => (root || document).querySelector(selector);
  const $$ = (selector, root) =>
    Array.from((root || document).querySelectorAll(selector));

  const state = {
    threshold: 0,
  };

  const LAYOUT = [
    {
      title: "Card & Spend",
      note: "Tier, fees, and annual economics",
      fields: [
        "card_tier",
        "tenure_months",
        "annual_fee_usd",
        "monthly_spend_usd",
        "annual_spend_usd",
      ],
    },
    {
      title: "Costs & Balance",
      note: "Rewards, service, and revolving balance",
      fields: [
        "rewards_cost_usd",
        "service_cost_usd",
        "statement_balance_usd",
        "utilization_pct",
        "delinquency_rate",
      ],
    },
    {
      title: "Risk & Profile",
      note: "Credit and engagement signals",
      fields: [
        "credit_score",
        "income_usd",
        "age",
        "gender",
        "region",
        "acquisition_channel",
        "has_premium_rewards",
        "travel_spend_pct",
        "dining_spend_pct",
        "grocery_spend_pct",
        "digital_engagement_score",
        "support_contacts_last_12m",
        "complaint_flag",
      ],
    },
  ];

  function setText(selector, value) {
    $$(selector).forEach((element) => {
      element.textContent = value;
    });
  }

  function hydrateMetadata() {
    if (!model) return;
    const metrics = model.metadata.metrics;
    setText(".js-threshold", model.metadata.threshold.toFixed(3));
    setText(".js-model-name", model.metadata.selectedModel);
    setText(".js-roc", metrics.rocAuc.toFixed(4));
    setText(".js-pr", metrics.prAuc.toFixed(4));
    const stamp = $("#modelStamp");
    if (stamp) {
      stamp.textContent = `${model.metadata.selectedModel} · ${new Date(
        model.metadata.trainedAt
      ).toLocaleDateString("en-US", { year: "numeric", month: "short", day: "numeric" })}`;
    }
  }

  function buildField(name) {
    const field = document.createElement("div");
    field.className = "field";
    const numericMeta = new Map((model.numeric || []).map((f) => [f.name, f]));
    const categoricalMeta = new Map(
      (model.categorical || []).map((f) => [f.name, f])
    );
    const meta = numericMeta.get(name) || categoricalMeta.get(name);
    if (!meta) return field;

    const label = document.createElement("label");
    label.setAttribute("for", `field-${name}`);
    label.textContent = meta.name.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
    field.appendChild(label);

    if (numericMeta.has(name)) {
      const input = document.createElement("input");
      input.className = "input";
      input.type = "number";
      input.id = `field-${name}`;
      input.name = name;
      input.step = "any";
      input.inputMode = "decimal";
      input.required = true;
      field.appendChild(input);
    } else if (categoricalMeta.has(name)) {
      const select = document.createElement("select");
      select.className = "select";
      select.id = `field-${name}`;
      select.name = name;
      select.required = true;
      Object.keys(categoricalMeta.get(name).coefficients).forEach((value) => {
        const option = document.createElement("option");
        option.value = value;
        option.textContent = value;
        select.appendChild(option);
      });
      field.appendChild(select);
    }
    return field;
  }

  function renderForm() {
    const body = $("#formBody");
    if (!body) return;
    body.replaceChildren();
    LAYOUT.forEach((group) => {
      const section = document.createElement("section");
      section.className = "form-group";
      const head = document.createElement("div");
      head.className = "form-group-head";
      const title = document.createElement("h4");
      title.textContent = group.title;
      const note = document.createElement("span");
      note.textContent = group.note;
      head.append(title, note);
      const grid = document.createElement("div");
      grid.className = "field-grid";
      group.fields.forEach((name) => {
        grid.appendChild(buildField(name));
      });
      section.append(head, grid);
      body.appendChild(section);
    });
  }

  function exampleById(id) {
    return model.examples.find((example) => example.id === id) || model.examples[0];
  }

  function setFormValues(raw) {
    for (const name of scorer.RAW_FEATURE_COLUMNS) {
      const input = $(`#field-${name}`);
      if (!input) continue;
      if (input.tagName === "SELECT") {
        const value = String(raw[name] == null ? "" : raw[name]);
        if (!Array.from(input.options).some((option) => option.value === value)) {
          input.add(new Option(value, value));
        }
        input.value = value;
      } else {
        input.value = raw[name] == null ? "" : String(raw[name]);
      }
    }
    updateScore();
  }

  function loadExample(id) {
    const example = exampleById(id);
    if (example) setFormValues(example.raw);
  }

  function numericInputValue(name) {
    const input = $(`#field-${name}`);
    if (!input) return NaN;
    const value = input.value.trim();
    return value === "" ? NaN : Number(value);
  }

  function readFormValues() {
    const raw = {};
    for (const name of scorer.RAW_FEATURE_COLUMNS) {
      const input = $(`#field-${name}`);
      if (!input) continue;
      raw[name] = input.value;
    }
    return raw;
  }

  function hideError() {
    const box = $("#resultError");
    if (box) {
      box.hidden = true;
      box.textContent = "";
    }
  }

  function showError(message) {
    const box = $("#resultError");
    if (box) {
      box.hidden = false;
      box.textContent = message;
    }
  }

  function renderResult(raw, result) {
    const probability = $("#probabilityValue");
    if (probability) {
      probability.replaceChildren(
        document.createTextNode((result.rawProbability * 100).toFixed(1)),
        (() => {
          const small = document.createElement("small");
          small.textContent = "%";
          return small;
        })()
      );
    }

    const badge = $("#riskBadge");
    if (badge) {
      badge.textContent = `${result.riskTier} risk`;
      badge.className = "risk-badge risk-" + result.riskTier.toLowerCase();
    }

    const decision = $("#decisionText");
    if (decision) {
      decision.replaceChildren();
      const strong = document.createElement("strong");
      strong.textContent = result.flagged
        ? "Flagged for migration review."
        : "Below the migration threshold.";
      const text = document.createTextNode(
        result.flagged
          ? ` The churn score is at or above the ${model.metadata.threshold.toFixed(3)} threshold.`
          : ` The churn score is below the ${model.metadata.threshold.toFixed(3)} threshold; maintain standard engagement.`
      );
      decision.append(strong, text);
    }

    const rec = result.recommendation;
    const recBox = $("#recommendationBox");
    if (recBox) {
      recBox.replaceChildren();
      const tierRow = document.createElement("div");
      tierRow.className = "tier-row";
      tierRow.appendChild(
        Object.assign(document.createElement("span"), {
          textContent: "Current tier: ",
          className: "label",
        })
      );
      tierRow.appendChild(
        Object.assign(document.createElement("span"), {
          textContent: rec.current_tier,
          className: "tier-badge",
        })
      );
      tierRow.appendChild(
        Object.assign(document.createElement("span"), {
          textContent: " → Optimal tier: ",
          className: "label",
        })
      );
      tierRow.appendChild(
        Object.assign(document.createElement("span"), {
          textContent: rec.optimal_tier,
          className: "tier-badge recommended",
        })
      );
      recBox.appendChild(tierRow);

      const liftRow = document.createElement("div");
      liftRow.className = "margin-row";
      liftRow.appendChild(
        Object.assign(document.createElement("span"), {
          textContent: "Expected annual margin lift: ",
          className: "label",
        })
      );
      liftRow.appendChild(
        Object.assign(document.createElement("span"), {
          textContent: scorer.formatMoneyCompact(rec.expected_margin_lift),
          className: "value",
        })
      );
      recBox.appendChild(liftRow);

      const actionRow = document.createElement("div");
      actionRow.className = "action-row";
      actionRow.appendChild(
        Object.assign(document.createElement("span"), {
          textContent: "Recommended action: ",
          className: "label",
        })
      );
      const actionText = document.createElement("span");
      actionText.textContent = rec.recommended_action;
      actionText.className = `action-${rec.action_type}`;
      actionRow.appendChild(actionText);
      recBox.appendChild(actionRow);

      const priorityRow = document.createElement("div");
      priorityRow.className = "priority-row";
      priorityRow.appendChild(
        Object.assign(document.createElement("span"), {
          textContent: "Migration priority: ",
          className: "label",
        })
      );
      const priorityBar = document.createElement("div");
      priorityBar.className = "priority-bar";
      const priorityFill = document.createElement("div");
      priorityFill.className = "priority-fill";
      priorityFill.style.width = `${result.priority}%`;
      priorityBar.appendChild(priorityFill);
      priorityRow.appendChild(priorityBar);
      const priorityVal = document.createElement("span");
      priorityVal.textContent = `${result.priority}/100`;
      priorityVal.className = "priority-value";
      priorityRow.appendChild(priorityVal);
      recBox.appendChild(priorityRow);
    }

    const marginList = $("#marginList");
    if (marginList) {
      marginList.replaceChildren();
      Object.entries(rec.tier_margins).forEach(([tier, margin]) => {
        const row = document.createElement("div");
        row.className = "margin-item";
        row.appendChild(
          Object.assign(document.createElement("span"), { textContent: tier })
        );
        row.appendChild(
          Object.assign(document.createElement("span"), {
            textContent: scorer.formatMoneyCompact(margin),
            className: "margin-value",
          })
        );
        marginList.appendChild(row);
      });
    }

    hideError();
  }

  function updateScore() {
    try {
      const raw = readFormValues();
      const result = scorer.scoreRaw(raw, model);
      renderResult(raw, result);
    } catch (error) {
      const probability = $("#probabilityValue");
      if (probability) probability.textContent = "—";
      showError(error.message || "The score could not be calculated.");
    }
  }

  /* Threshold lab ------------------------------------------------------- */
  function setupThreshold() {
    const slider = $("#thresholdSlider");
    if (!slider) return;
    slider.value = model.metadata.threshold.toFixed(3);
    state.threshold = model.metadata.threshold;
    renderThreshold(model.metadata.threshold);
    slider.addEventListener("input", () => {
      renderThreshold(Number(slider.value));
    });
  }

  function renderThreshold(threshold) {
    state.threshold = threshold;
    const metrics = scorer.thresholdMetrics(model, threshold);
    setText("#thresholdValue", threshold.toFixed(3));
    const chip = $(".threshold-chip");
    if (chip) {
      chip.textContent =
        Math.abs(threshold - model.metadata.threshold) < 0.0005
          ? "selected policy"
          : "custom policy";
    }
    setText("#labContacts", metrics.contacts.toLocaleString("en-US"));
    setText("#labAlertRate", scorer.formatPct(metrics.alertRate));
    setText("#labPrecision", scorer.formatPct(metrics.precision));
    setText("#labRecall", scorer.formatPct(metrics.recall));
    setText("#labF1", metrics.f1.toFixed(3));
  }

  /* Batch --------------------------------------------------------------- */
  function setupBatch() {
    const loadSample = $("#loadSample");
    if (loadSample) {
      loadSample.addEventListener("click", () => {
        // Build a small CSV from examples
        const headers = scorer.RAW_FEATURE_COLUMNS.join(",");
        const rows = model.examples
          .map((example, index) => {
            const values = scorer.RAW_FEATURE_COLUMNS.map(
              (col) => example.raw[col] == null ? "" : example.raw[col]
            );
            return values.join(",");
          })
          .join("\r\n");
        ingestCsv(headers + "\r\n" + rows, "sample_cards.csv");
      });
    }
  }

  function ingestCsv(text, fileName) {
    try {
      const lines = text.split(/\r?\n/).filter((line) => line.trim() !== "");
      if (lines.length < 2) throw new Error("CSV needs header and data rows.");
      const headers = lines[0].split(",").map((h) => h.trim());
      const missing = scorer.RAW_FEATURE_COLUMNS.filter(
        (c) => !headers.includes(c)
      );
      if (missing.length > 0) {
        throw new Error(`Missing required columns: ${missing.join(", ")}`);
      }
      const results = [];
      for (let i = 1; i < lines.length; i += 1) {
        const values = lines[i].split(",");
        const raw = {};
        headers.forEach((h, idx) => {
          raw[h] = values[idx] == null ? "" : values[idx];
        });
        const result = scorer.scoreRaw(raw, model);
        results.push({
          customerID: raw.customerID || `row-${i}`,
          churn_probability: result.probability,
          risk_tier: result.riskTier,
          current_tier: result.recommendation.current_tier,
          optimal_tier: result.recommendation.optimal_tier,
          expected_margin_lift: result.recommendation.expected_margin_lift,
          priority: result.priority,
        });
      }
      results.sort((a, b) => b.priority - a.priority);
      renderBatch(results, fileName);
    } catch (error) {
      showBatchError(error.message || "The CSV could not be scored.");
    }
  }

  function renderBatch(results, fileName) {
    const body = $("#batchBody");
    if (body) {
      body.replaceChildren();
      results.slice(0, 25).forEach((row) => {
        const tr = document.createElement("tr");
        const values = [
          row.customerID,
          `${(row.churn_probability * 100).toFixed(1)}%`,
          row.risk_tier,
          row.current_tier,
          row.optimal_tier,
          scorer.formatMoneyCompact(row.expected_margin_lift),
          row.priority,
        ];
        values.forEach((value) => {
          const cell = document.createElement("td");
          cell.textContent = value;
          tr.appendChild(cell);
        });
        body.appendChild(tr);
      });
    }
    setText("#batchRows", results.length.toLocaleString("en-US"));
    const status = $("#batchStatus");
    if (status) {
      status.textContent = `Scored ${results.length.toLocaleString("en-US")} customers from ${fileName}.`;
      status.className = "batch-status is-success";
    }
  }

  function showBatchError(message) {
    const status = $("#batchStatus");
    if (status) {
      status.textContent = message;
      status.className = "batch-status is-error";
    }
  }

  function init() {
    if (!model || !scorer) {
      showError(
        "The exported model data could not be loaded. Open the page through the deployed site or a local web server."
      );
      return;
    }
    renderForm();
    hydrateMetadata();
    setupThreshold();
    setupBatch();
    loadExample("gold-upsell");
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
