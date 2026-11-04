/* Read-only view reductions. Forecasting continues in local-demo-core.mjs. */
const mean = (values) => values.reduce((sum, value) => sum + value, 0) / values.length;

export function routingAt(result, indicator, hour) {
  if (!Number.isInteger(indicator) || indicator < 0 || indicator > 3 || !Number.isInteger(hour) || hour < 0 || hour > 23)
    throw new RangeError('Invalid forecast position');
  const members = result.members;
  return Array.from({ length: 8 }, (_, expert) => {
    const available = members[0].availability[hour][indicator][expert];
    if (members.some(m => m.availability[hour][indicator][expert] !== available))
      throw new Error('Members disagree on expert availability');
    return {
      available,
      support: mean(members.map(m => m.reliability[hour][indicator][expert])),
      value: available ? mean(members.map(m => Math.fround(Math.max(1e-4, Math.expm1(m.expert_values_log[hour][indicator][expert]))))) : null,
      weight: mean(members.map(m => m.attention[hour][indicator][expert])),
    };
  });
}

export function ensembleEnvelope(result) {
  const reduce = (key) => Array.from({ length: 24 }, (_, h) => Array.from({ length: 4 }, (_, q) => mean(result.members.map(m => m[key][h][q]))));
  return { lower: reduce('lower_envelope'), upper: reduce('upper_envelope') };
}
