/* Amex Card Tier Migration browser scorer.
 *
 * Mirrors the fitted scikit-learn pipeline exported in scorer_data.js.
 * No network calls, no third-party dependencies. The CommonJS export is
 * intentional: tests/browser_scorer.test.js exercises the same functions
 * the page uses.
 */
(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) {
    module.exports = api;
  } else {
    root.CardScorer = api;
  }
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  const TIERS = ["No-Fee", "Silver", "Gold", "Platinum", "Centurion"];

  const RAW_FEATURE_COLUMNS = [
    "customerID",
    "card_tier",
    "tenure_months",
    "annual_fee_usd",
    "monthly_spend_usd",
    "annual_spend_usd",
    "rewards_cost_usd",
    "service_cost_usd",
    "statement_balance_usd",
    "utilization_pct",
    "delinquency_rate",
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
    "Churn",
  ];

  const INTERCHANGE_RATE = 0.018;
  const INTEREST_RATE = 0.12;

  function clamp(value, minimum, maximum) {
    return Math.min(maximum, Math.max(minimum, value));
  }

  function roundTo(value, digits) {
    const factor = 10 ** digits;
    return Math.round((value + Number.EPSILON) * factor) / factor;
  }

  function sigmoid(logit) {
    if (logit >= 0) {
      const z = Math.exp(-logit);
      return 1 / (1 + z);
    }
    const z = Math.exp(logit);
    return z / (1 + z);
  }

  function numericValue(value) {
    if (value === null || value === undefined || String(value).trim() === "") {
      return NaN;
    }
    const number = Number(value);
    return Number.isFinite(number) ? number : NaN;
  }

  function deriveFeatures(raw, artifact) {
    const tenure = numericValue(raw.tenure_months);
    const monthlySpend = numericValue(raw.monthly_spend_usd);
    const annualFee = numericValue(raw.annual_fee_usd);
    const annualSpend = numericValue(raw.annual_spend_usd);
    const statementBalance = numericValue(raw.statement_balance_usd);
    const utilization = numericValue(raw.utilization_pct);

    if (!Number.isFinite(tenure)) throw new Error("Tenure must be a number.");
    if (tenure < 0) throw new Error("Tenure cannot be negative.");
    if (!Number.isFinite(annualSpend)) throw new Error("Annual spend must be a number.");
    if (annualSpend < 0) throw new Error("Annual spend cannot be negative.");

    const spendToFeeRatio =
      annualFee > 0 ? annualSpend / annualFee : 0;
    const spendPerTenure =
      tenure > 0 ? annualSpend / tenure : 0;

    const rewardsCost = numericValue(raw.rewards_cost_usd);
    const serviceCost = numericValue(raw.service_cost_usd);

    const revenue =
      annualFee +
      annualSpend * INTERCHANGE_RATE +
      statementBalance * INTEREST_RATE * 0.5;
    const marginProxy = revenue - rewardsCost - serviceCost;

    const isPremiumTier =
      raw.card_tier === "Platinum" || raw.card_tier === "Centurion" ? 1 : 0;
    const highValueSegment = annualSpend >= 60000 ? 1 : 0;

    return {
      card_tier: raw.card_tier == null ? "" : String(raw.card_tier),
      tenure_months: tenure,
      annual_fee_usd: annualFee,
      monthly_spend_usd: monthlySpend,
      annual_spend_usd: annualSpend,
      rewards_cost_usd: rewardsCost,
      service_cost_usd: serviceCost,
      statement_balance_usd: statementBalance,
      utilization_pct: utilization,
      delinquency_rate: numericValue(raw.delinquency_rate),
      credit_score: numericValue(raw.credit_score),
      income_usd: numericValue(raw.income_usd),
      age: numericValue(raw.age),
      gender: String(raw.gender == null ? "" : raw.gender),
      region: String(raw.region == null ? "" : raw.region),
      acquisition_channel: String(
        raw.acquisition_channel == null ? "" : raw.acquisition_channel
      ),
      has_premium_rewards: String(
        raw.has_premium_rewards == null ? "" : raw.has_premium_rewards
      ),
      travel_spend_pct: numericValue(raw.travel_spend_pct),
      dining_spend_pct: numericValue(raw.dining_spend_pct),
      grocery_spend_pct: numericValue(raw.grocery_spend_pct),
      digital_engagement_score: numericValue(raw.digital_engagement_score),
      support_contacts_last_12m: numericValue(
        raw.support_contacts_last_12m
      ),
      complaint_flag: String(raw.complaint_flag == null ? "" : raw.complaint_flag),
      spend_to_fee_ratio: spendToFeeRatio,
      spend_per_tenure: spendPerTenure,
      margin_proxy: marginProxy,
      is_premium_tier: isPremiumTier,
      high_value_segment: highValueSegment,
    };
  }

  function scoreFeatures(features, artifact) {
    let logit = artifact.intercept;

    for (const feature of artifact.numeric) {
      let value = numericValue(features[feature.name]);
      if (!Number.isFinite(value)) {
        value = feature.median;
      }
      logit += ((value - feature.mean) / feature.scale) * feature.coefficient;
    }

    for (const feature of artifact.categorical) {
      const value = String(features[feature.name]);
      if (Object.prototype.hasOwnProperty.call(feature.coefficients, value)) {
        logit += feature.coefficients[value];
      }
      // Unknown levels ignored, matching OneHotEncoder(handle_unknown="ignore").
    }

    const rawProbability = sigmoid(logit);
    const probability = roundTo(rawProbability, 4);

    const bands = artifact.metadata.riskBands;
    let band = bands[bands.length - 1];
    for (const b of bands) {
      if (rawProbability >= b.min && rawProbability < b.max) {
        band = b;
        break;
      }
    }

    const flagged = rawProbability >= artifact.metadata.threshold;

    // Margin economics per tier
    const tierMargins = {};
    for (const tier of TIERS) {
      tierMargins[tier] = computeAnnualMargin(features, tier, artifact);
    }
    const currentTier = features.card_tier || "Gold";
    const optimalTier = Object.keys(tierMargins).reduce(
      (best, tier) => (tierMargins[tier] > tierMargins[best] ? tier : best),
      TIERS[0]
    );
    const marginLift = tierMargins[optimalTier] - (tierMargins[currentTier] || 0);

    const recommendation = {
      current_tier: currentTier,
      optimal_tier: optimalTier,
      current_margin: roundTo(tierMargins[currentTier] || 0, 2),
      optimal_margin: roundTo(tierMargins[optimalTier], 2),
      expected_margin_lift: roundTo(marginLift, 2),
      churn_probability: probability,
      tier_margins: Object.fromEntries(
        Object.entries(tierMargins).map(([t, m]) => [t, roundTo(m, 2)])
      ),
    };

    const priority = clamp(
      Math.round(
        (marginLift / 500) * 40 +
          rawProbability * 30 +
          (1 - features.digital_engagement_score) * 20
      ),
      0,
      100
    );

    return {
      logit,
      rawProbability,
      probability,
      riskTier: band.name,
      riskColor: band.color,
      flagged,
      recommendation,
      priority,
    };
  }

  function scoreRaw(raw, artifact) {
    const features = deriveFeatures(raw, artifact);
    return {
      features,
      ...scoreFeatures(features, artifact),
    };
  }

  function computeAnnualMargin(features, tier, artifact) {
    const tierAnnualFee =
      artifact.metadata.tierAnnualFee && artifact.metadata.tierAnnualFee[tier]
        ? artifact.metadata.tierAnnualFee[tier]
        : 0;
    const annualSpend = features.annual_spend_usd;
    const statementBalance = features.statement_balance_usd;
    const delinquency = features.delinquency_rate;
    const rewardsCost = features.rewards_cost_usd;
    const serviceCost = features.service_cost_usd;

    const revenue =
      tierAnnualFee +
      annualSpend * INTERCHANGE_RATE +
      statementBalance * INTEREST_RATE * 0.5;
    const expectedLoss = statementBalance * delinquency;
    return revenue - rewardsCost - serviceCost - expectedLoss;
  }

  function thresholdMetrics(artifact, threshold) {
    const holdout = artifact.holdout;
    let tp = 0;
    let fp = 0;
    let fn = 0;
    let tn = 0;

    for (let i = 0; i < holdout.probabilities.length; i += 1) {
      const probability = holdout.probabilities[i];
      const actual = holdout.labels.charAt(i) === "1";
      const contacted = probability >= threshold;
      if (contacted && actual) tp += 1;
      else if (contacted && !actual) fp += 1;
      else if (!contacted && actual) fn += 1;
      else tn += 1;
    }

    const contacts = tp + fp;
    const precision = contacts > 0 ? tp / contacts : 0;
    const recall = tp + fn > 0 ? tp / (tp + fn) : 0;
    const f1 =
      precision + recall > 0
        ? (2 * precision * recall) / (precision + recall)
        : 0;

    return {
      threshold,
      contacts,
      alertRate: contacts / Math.max(1, holdout.n),
      precision,
      recall,
      f1,
      tp,
      fp,
      fn,
      tn,
    };
  }

  function tradeoffCurve(artifact, minimum, maximum, steps) {
    const count = steps || 81;
    const points = [];
    for (let i = 0; i < count; i += 1) {
      const threshold =
        minimum + ((maximum - minimum) * i) / Math.max(1, count - 1);
      points.push(thresholdMetrics(artifact, threshold));
    }
    return points;
  }

  function formatMoney(value) {
    return Number(value).toLocaleString("en-US", {
      style: "currency",
      currency: "USD",
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    });
  }

  function formatMoneyCompact(value) {
    const number = Number(value);
    if (Math.abs(number) >= 1000000) {
      return `$${(number / 1000000).toFixed(2)}M`;
    }
    if (Math.abs(number) >= 1000) {
      return `$${(number / 1000).toFixed(1)}k`;
    }
    return formatMoney(number);
  }

  function formatPct(value, digits) {
    return `${(Number(value) * 100).toFixed(digits == null ? 1 : digits)}%`;
  }

  return {
    TIERS,
    RAW_FEATURE_COLUMNS,
    INTERCHANGE_RATE,
    INTEREST_RATE,
    clamp,
    roundTo,
    sigmoid,
    deriveFeatures,
    scoreFeatures,
    scoreRaw,
    computeAnnualMargin,
    thresholdMetrics,
    tradeoffCurve,
    formatMoney,
    formatMoneyCompact,
    formatPct,
  };
});
