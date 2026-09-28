const heroes = JSON.parse(document.getElementById('hero-data').textContent);
const byId = new Map(heroes.map(hero => [hero.id, hero]));
const draft = {allies: [], enemies: [], banned: []};
let active = 'allies';
let revision = 0;
let pending = false;
const status = document.getElementById('status');
const recommend = document.getElementById('recommend');
const results = document.getElementById('results');
const grid = document.getElementById('hero-grid');

function portrait(hero) {
  const img = document.createElement('img');
  img.src = hero.image;
  img.alt = hero.name;
  return img;
}

function changed() {
  revision++;
  results.hidden = true;
  status.textContent = 'Select 4 heroes for each team.';
  render();
}

function render() {
  for (const side of ['allies', 'enemies']) {
    const team = document.getElementById(side);
    team.classList.toggle('active', side === active);
    team.querySelector('.team-target').setAttribute('aria-pressed', side === active);
    team.querySelector('span').textContent = `${draft[side].length} / 4`;
    const slots = team.querySelector('.slots');
    slots.replaceChildren();
    for (let index = 0; index < 4; index++) {
      const slot = document.createElement('button');
      slot.className = 'slot';
      const hero = byId.get(draft[side][index]);
      slot.title = hero ? `Remove ${hero.name}` : `Select for ${side}`;
      slot.setAttribute('aria-label', slot.title);
      if (hero) slot.append(portrait(hero));
      else slot.textContent = '+';
      slot.onclick = () => {
        active = side;
        if (hero) { draft[side].splice(index, 1); changed(); }
        else render();
      };
      slots.append(slot);
    }
  }
  recommend.disabled = pending || draft.allies.length !== 4 || draft.enemies.length !== 4;
  for (const button of grid.querySelectorAll('.hero')) {
    const id = Number(button.dataset.id);
    const side = ['allies', 'enemies'].find(team => draft[team].includes(id));
    const banned = draft.banned.includes(id);
    button.className = `hero ${side || ''} ${banned ? 'banned' : ''}`;
    button.querySelector('.badge').textContent = banned ? '✕' : side === 'allies' ? 'ALLY' : side === 'enemies' ? 'ENEMY' : '';
    button.setAttribute('aria-label', `${byId.get(id).name}${banned ? ', banned' : side ? `, ${side}` : ''}`);
  }
}

for (const [attribute, label] of [['str', 'Strength'], ['agi', 'Agility'], ['int', 'Intelligence'], ['all', 'Universal']]) {
  const section = document.createElement('section');
  const heading = document.createElement('h2');
  heading.textContent = label;
  section.append(heading);
  const group = document.createElement('div');
  group.className = 'portraits';
  for (const hero of heroes.filter(hero => hero.attribute === attribute)) {
    const button = document.createElement('button');
    button.className = 'hero';
    button.dataset.id = hero.id;
    button.title = hero.name;
    button.append(portrait(hero));
    const badge = document.createElement('span');
    badge.className = 'badge';
    button.append(badge);
    button.onclick = () => {
      if (draft.banned.includes(hero.id)) { status.textContent = 'Unban this hero before selecting.'; return; }
      if (draft.allies.includes(hero.id) || draft.enemies.includes(hero.id)) { status.textContent = 'Already selected. Click its team slot to remove.'; return; }
      if (draft[active].length === 4) { status.textContent = 'This team is full. Select the other team or remove a hero.'; return; }
      draft[active].push(hero.id);
      changed();
    };
    button.oncontextmenu = event => {
      event.preventDefault();
      if (draft.banned.includes(hero.id)) draft.banned = draft.banned.filter(id => id !== hero.id);
      else {
        draft.allies = draft.allies.filter(id => id !== hero.id);
        draft.enemies = draft.enemies.filter(id => id !== hero.id);
        draft.banned.push(hero.id);
      }
      changed();
    };
    group.append(button);
  }
  section.append(group);
  grid.append(section);
}

document.querySelectorAll('.team').forEach(team => {
  team.onclick = () => { active = team.id; render(); };
});
document.getElementById('search').oninput = event => {
  const term = event.target.value.trim().toLowerCase();
  grid.querySelectorAll('.hero').forEach(button => {
    button.hidden = !byId.get(Number(button.dataset.id)).name.toLowerCase().includes(term);
  });
  grid.querySelectorAll('section').forEach(section => {
    section.hidden = !section.querySelector('.hero:not([hidden])');
  });
};
document.getElementById('clear').onclick = () => {
  draft.allies = []; draft.enemies = []; draft.banned = [];
  active = 'allies'; changed();
};
document.getElementById('example').onclick = () => {
  draft.allies = [2, 5, 25, 86]; draft.enemies = [14, 26, 74, 99]; draft.banned = [];
  changed();
};
recommend.onclick = async () => {
  const requestedRevision = revision;
  pending = true;
  render();
  status.textContent = 'Ranking carry candidates…';
  try {
    const response = window.recommendCarry ? await window.recommendCarry(draft) : await fetch('/api/recommend', {
      method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(draft)
    });
    const data = await response.json();
    if (revision !== requestedRevision) return;
    if (!response.ok) throw new Error(data.error || 'Recommendation failed.');
    const list = results.querySelector('ol');
    list.replaceChildren();
    for (const [index, item] of data.recommendations.entries()) {
      const row = document.createElement('li');
      const name = document.createElement('span');
      name.textContent = `${index + 1}. ${item.hero_name}`;
      const score = document.createElement('strong');
      score.textContent = item.model_score.toFixed(3);
      row.append(portrait(byId.get(item.hero_id)), name, score);
      list.append(row);
    }
    results.hidden = false;
    status.textContent = data.recommendations.length ? 'Ranked from the final model’s candidate pool.' : 'No available carry candidates. Unban heroes to continue.';
  } catch (error) {
    if (revision === requestedRevision) status.textContent = error.message;
  } finally {
    pending = false;
    render();
  }
};
render();
