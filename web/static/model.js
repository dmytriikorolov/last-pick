// Runs the exported checkpoint in the browser for the hosted static site.
(function () {
  let loaded;
  const mean = rows => rows[0].map((_, i) => rows.reduce((sum, row) => sum + row[i], 0) / rows.length);
  const rate = (wins, games) => (wins + 50) / (games + 100);
  const reliability = games => games / (games + 100);

  function pairs(stats, candidate, others, allied) {
    const table = stats[allied ? 'synergy' : 'matchups'];
    const profile = stats.heroes[candidate] || Array(14).fill(0);
    const candidateRate = rate(profile[1], profile[0]);
    const values = others.map(hero => {
      const key = allied ? [candidate, hero].sort((a, b) => a - b).join(':') : `${candidate}:${hero}`;
      const [games, wins] = table[key] || [0, 0];
      const other = stats.heroes[hero] || Array(14).fill(0);
      const reference = allied ? (candidateRate + rate(other[1], other[0])) / 2 : candidateRate;
      return [games ? (wins / games - reference) * reliability(games) : 0,
        Math.log1p(games), reliability(games)];
    });
    const averages = mean(values);
    const deltas = values.map(row => row[0]);
    return [averages[0], allied ? Math.max(...deltas) : Math.min(...deltas), averages[1], averages[2]];
  }

  function historyValues(bundle, candidate, allies, enemies) {
    const profile = bundle.stats.heroes[candidate] || Array(14).fill(0);
    const values = [rate(profile[1], profile[0]), Math.log1p(profile[0])];
    for (let i = 0; i < 3; i++) values.push(rate(profile[3 + 2 * i], profile[2 + 2 * i]));
    for (let i = 0; i < 3; i++) values.push(reliability(profile[2 + 2 * i]));
    for (let i = 0; i < 3; i++) values.push(profile[9 + 2 * i] ? profile[8 + 2 * i] / profile[9 + 2 * i] : 0);
    values.push([9, 11, 13].reduce((sum, i) => sum + reliability(profile[i]), 0) / 3);
    values.push(...pairs(bundle.stats, candidate, allies, true), ...pairs(bundle.stats, candidate, enemies, false));
    // Python forms a float32 tensor before scaling historical inputs.
    return bundle.indices.map((index, i) => Math.fround(
      Math.fround(Math.fround(values[index]) - Math.fround(bundle.mean[i])) / Math.fround(bundle.scale[i])));
  }

  function score(model, candidate, allies, enemies, mapping) {
    const carry = model.embedding[mapping.get(candidate)];
    const ally = mean(allies.map(hero => model.embedding[mapping.get(hero)]));
    const enemy = mean(enemies.map(hero => model.embedding[mapping.get(hero)]));
    let values = [...carry, ...ally, ...enemy,
      ...carry.map((value, i) => value * ally[i]), ...carry.map((value, i) => value * enemy[i])];
    if (model.use_static) values.push(...model.hero_features[mapping.get(candidate)],
      ...mean(allies.map(hero => model.hero_features[mapping.get(hero)])),
      ...mean(enemies.map(hero => model.hero_features[mapping.get(hero)])));
    if (model.history) values.push(...historyValues(model.history, candidate, allies, enemies));
    for (const [index, layer] of model.layers.entries()) {
      const input = values;
      values = layer.weight.map((row, i) => {
        const value = row.reduce((sum, weight, j) => sum + weight * input[j], layer.bias[i]);
        return index === model.layers.length - 1 ? value : Math.max(0, value);
      });
    }
    return 1 / (1 + Math.exp(-values[0]));
  }

  function rank(model, draft) {
    const mapping = new Map(model.hero_ids.map((id, i) => [id, i + 1]));
    const visible = new Set([...draft.allies, ...draft.enemies, ...draft.banned]);
    return model.candidate_pool.filter(id => !visible.has(id)).map(id => ({
      hero_id: id, hero_name: model.names[id],
      model_score: score(model, id, draft.allies, draft.enemies, mapping)
    })).sort((a, b) => b.model_score - a.model_score || a.hero_id - b.hero_id).slice(0, 5);
  }

  if (typeof module !== 'undefined') module.exports = {rank};
  if (typeof window !== 'undefined') window.recommendCarry = async draft => {
    if (!loaded) loaded = fetch('/static/model.json').then(response => {
      if (!response.ok) throw new Error('Could not load the recommendation model.');
      return response.json();
    }).catch(error => { loaded = null; throw error; });
    const recommendations = rank(await loaded, draft);
    return {ok: true, json: async () => ({recommendations})};
  };
})();
