(() => {
  const byElement = id => document.getElementById(id);
  const heroes = JSON.parse(byElement('hero-data').textContent);
  const byId = new Map(heroes.map(hero => [hero.id, hero]));
  const supported = new Set(JSON.parse(byElement('candidate-data').textContent));
  const groups = ['allies', 'enemies', 'banned'];
  const teamNames = {allies: 'your team', enemies: 'enemy team'};
  const favoritesKey = 'lastpick.favorites.v1';
  const poolFilterKey = 'lastpick.pool-filter.v1';
  const draftsKey = 'lastpick.drafts.v1';
  const status = byElement('status');
  const savedStatus = byElement('saved-status');
  const recommend = byElement('recommend');
  const results = byElement('results');
  const grid = byElement('hero-grid');
  const poolFilter = byElement('carry-pool');
  const nameDialog = byElement('save-dialog');
  const draftName = byElement('draft-name');
  const roster = new Map();
  const slots = new Map();
  const favoriteIds = readStorage(favoritesKey);
  const favorites = new Set(Array.isArray(favoriteIds) ? favoriteIds.filter(id => byId.has(id)) : []);
  const savedFilter = readStorage(poolFilterKey);
  let poolPreference = typeof savedFilter === 'boolean' ? savedFilter : null;
  let historyPool = [];
  const storedDrafts = readStorage(draftsKey);
  let savedDrafts = Array.isArray(storedDrafts) ? storedDrafts.filter(isSavedDraft) : [];
  let draft = {allies: [], enemies: [], banned: []};
  let active = 'allies';
  let mode = 'pick';
  let revision = 0;
  let pending = false;
  let editingDraftId = null;

  function readStorage(key) {
    try {
      return JSON.parse(localStorage.getItem(key));
    } catch {
      return null;
    }
  }

  function writeStorage(key, value) {
    try {
      localStorage.setItem(key, JSON.stringify(value));
      return true;
    } catch {
      return false;
    }
  }

  function poolIds() {
    return [...new Set([...historyPool, ...favorites])];
  }

  window.lastpickFavorites = {
    poolIds,
    setHistoryPool(ids) {
      historyPool = ids;
      poolFilter.checked = poolPreference ?? (historyPool.length > 0);
      changeDraft();
    },
    has: id => favorites.has(id),
    toggle(id) {
      if (!byId.has(id)) return;
      if (favorites.has(id)) favorites.delete(id);
      else favorites.add(id);
      const stored = writeStorage(favoritesKey, [...favorites]);
      window.dispatchEvent(new CustomEvent('lastpick:favorites', {detail: {stored}}));
      return favorites.has(id);
    }
  };

  function isDraft(value) {
    if (!value || !groups.every(group => Array.isArray(value[group]))) return false;
    const ids = groups.flatMap(group => value[group]);
    return value.allies.length <= 4 && value.enemies.length <= 4 &&
      ids.every(id => byId.has(id)) && new Set(ids).size === ids.length;
  }

  function isSavedDraft(value) {
    return value && typeof value.id === 'string' && typeof value.name === 'string' && isDraft(value.draft);
  }

  function copyDraft(value) {
    return {allies: [...value.allies], enemies: [...value.enemies], banned: [...value.banned]};
  }

  function portrait(hero) {
    const img = document.createElement('img');
    img.src = hero.image;
    img.alt = '';
    img.loading = 'lazy';
    return img;
  }

  function changeDraft(message = '') {
    revision++;
    results.hidden = true;
    status.textContent = message || (isComplete() ? 'Draft ready to rank.' : 'Add four heroes to each team.');
    renderDraft();
  }

  function isComplete() {
    return draft.allies.length === 4 && draft.enemies.length === 4;
  }

  function renderDraft() {
    for (const side of ['allies', 'enemies']) {
      const team = byElement(side);
      team.classList.toggle('active', side === active && mode === 'pick');
      team.querySelector('.team-target').setAttribute('aria-pressed', side === active && mode === 'pick');
      team.querySelector('.team-count').textContent = `${draft[side].length} / 4`;
      for (const [index, slot] of slots.get(side).entries()) {
        const hero = byId.get(draft[side][index]);
        slot.title = hero ? `Remove ${hero.name}` : `Empty slot for ${teamNames[side]}`;
        slot.setAttribute('aria-label', slot.title);
        slot.replaceChildren(hero ? portrait(hero) : '+');
        slot.classList.toggle('filled', Boolean(hero));
      }
    }
    byElement('pick-mode').setAttribute('aria-pressed', mode === 'pick');
    byElement('ban-mode').setAttribute('aria-pressed', mode === 'ban');
    byElement('picker-hint').textContent = mode === 'ban' ? 'Select a hero to ban or unban' : `Picking for ${teamNames[active]}`;
    recommend.disabled = pending || !isComplete();
    const hasDraft = groups.some(group => draft[group].length > 0);
    byElement('save-draft').disabled = !hasDraft;
    byElement('share-draft').disabled = !hasDraft;
    const favoriteCount = [...favorites].filter(id => supported.has(id)).length;
    byElement('pool-hint').textContent = historyPool.length
      ? `${poolIds().filter(id => supported.has(id)).length} supported heroes from your matches and favorites`
      : favorites.size
      ? `${favoriteCount} of ${favorites.size} favorites supported by the model`
      : 'Load My Dota to build a pool automatically, or star heroes.';
    for (const [id, {card, pick, star, badge}] of roster) {
      const side = ['allies', 'enemies'].find(team => draft[team].includes(id));
      const banned = draft.banned.includes(id);
      const selected = banned || Boolean(side);
      const favorite = favorites.has(id);
      const name = byId.get(id).name;
      card.dataset.state = banned ? 'banned' : side || '';
      badge.textContent = banned ? 'Banned' : side === 'allies' ? 'Ally' : side === 'enemies' ? 'Enemy' : '';
      badge.hidden = !selected;
      pick.setAttribute('aria-label', selected ? `${name}, ${badge.textContent}` : name);
      star.textContent = favorite ? '★' : '☆';
      star.title = historyPool.includes(id)
        ? `${favorite ? 'Unstar' : 'Star'} ${name}`
        : favorite ? `Remove ${name} from carry pool` : `Add ${name} to carry pool`;
      star.setAttribute('aria-label', star.title);
      star.setAttribute('aria-pressed', favorite);
    }
    renderBans();
  }

  function renderBans() {
    const list = byElement('ban-list');
    list.hidden = !draft.banned.length;
    const items = draft.banned.map(heroId => {
      const hero = byId.get(heroId);
      return button(`${hero.name} ×`, () => toggleBan(heroId), `Unban ${hero.name}`);
    });
    list.replaceChildren('Banned', ...items);
  }

  function selectHero(id) {
    if (mode === 'ban') {
      toggleBan(id);
      return;
    }
    if (draft.banned.includes(id)) {
      status.textContent = 'Unban this hero before picking.';
    } else if (draft.allies.includes(id) || draft.enemies.includes(id)) {
      status.textContent = 'Already picked. Select its team slot to remove it.';
    } else if (draft[active].length === 4) {
      status.textContent = 'This team is full. Switch teams or remove a hero.';
    } else {
      draft[active].push(id);
      changeDraft();
    }
  }

  function toggleBan(id) {
    if (draft.banned.includes(id)) draft.banned = draft.banned.filter(hero => hero !== id);
    else {
      draft.allies = draft.allies.filter(hero => hero !== id);
      draft.enemies = draft.enemies.filter(hero => hero !== id);
      draft.banned.push(id);
    }
    changeDraft();
  }

  function button(label, handler, ariaLabel) {
    const node = document.createElement('button');
    node.type = 'button';
    node.textContent = label;
    node.addEventListener('click', handler);
    if (ariaLabel) node.setAttribute('aria-label', ariaLabel);
    return node;
  }

  function buildTeams() {
    for (const side of ['allies', 'enemies']) {
      const team = byElement(side);
      team.querySelector('.team-target').addEventListener('click', () => {
        active = side;
        mode = 'pick';
        renderDraft();
      });
      const controls = Array.from({length: 4}, (_, index) => {
        const slot = button('+', () => {
          active = side;
          mode = 'pick';
          if (draft[side][index] !== undefined) {
            draft[side].splice(index, 1);
            changeDraft();
          } else {
            renderDraft();
          }
        });
        slot.className = 'slot';
        return slot;
      });
      slots.set(side, controls);
      team.querySelector('.slots').append(...controls);
    }
  }

  function buildRoster() {
    for (const [attribute, label] of [['str', 'Strength'], ['agi', 'Agility'], ['int', 'Intelligence'], ['all', 'Universal']]) {
      const section = document.createElement('section');
      const heading = document.createElement('h3');
      heading.textContent = label;
      const row = document.createElement('div');
      row.className = 'portraits';
      for (const hero of heroes.filter(item => item.attribute === attribute)) {
        const card = document.createElement('div');
        card.className = 'hero-card';
        const pick = button('', () => selectHero(hero.id));
        pick.className = 'hero-pick';
        pick.title = hero.name;
        const name = document.createElement('span');
        name.className = 'hero-name';
        name.textContent = hero.name;
        const badge = document.createElement('span');
        badge.className = 'hero-badge';
        badge.hidden = true;
        pick.append(portrait(hero), name, badge);
        pick.addEventListener('contextmenu', event => {
          event.preventDefault();
          toggleBan(hero.id);
        });
        const star = button('', () => window.lastpickFavorites.toggle(hero.id));
        star.className = 'favorite-button';
        roster.set(hero.id, {card, pick, star, badge});
        card.append(pick, star);
        row.append(card);
      }
      section.append(heading, row);
      grid.append(section);
    }
  }

  function updateSearch() {
    const term = byElement('search').value.trim().toLowerCase();
    for (const [id, {card}] of roster) card.hidden = !byId.get(id).name.toLowerCase().includes(term);
    let visible = 0;
    for (const section of grid.querySelectorAll('section')) {
      section.hidden = !section.querySelector('.hero-card:not([hidden])');
      if (!section.hidden) visible++;
    }
    byElement('no-heroes').hidden = visible > 0;
  }

  function openDraft(value, message) {
    draft = copyDraft(value);
    active = 'allies';
    mode = 'pick';
    changeDraft(message);
  }

  function showView() {
    const hash = window.location.hash;
    if (hash === '#main-content') return;
    const profile = hash === '#profile';
    byElement('draft-view').hidden = profile;
    byElement('profile-view').hidden = !profile;
    byElement(profile ? 'profile-link' : 'draft-link').setAttribute('aria-current', 'page');
    byElement(profile ? 'draft-link' : 'profile-link').removeAttribute('aria-current');
    if (!hash.startsWith('#draft?')) return;
    const params = new URLSearchParams(hash.slice(7));
    if (!groups.every(group => params.has(group))) {
      status.textContent = 'This draft link is incomplete.';
      return;
    }
    const data = Object.fromEntries(groups.map(group => {
      const raw = params.get(group);
      return [group, raw ? raw.split(',').map(part => /^[1-9]\d*$/.test(part) ? Number(part) : NaN) : []];
    }));
    if (isDraft(data)) openDraft(data, 'Shared draft opened.');
    else status.textContent = 'This draft link has invalid or duplicate heroes.';
  }

  function renderSaved(focusId) {
    const list = byElement('saved-drafts');
    list.replaceChildren();
    byElement('saved-empty').hidden = savedDrafts.length > 0;
    for (const item of savedDrafts) {
      const row = document.createElement('li');
      const title = button(item.name, () => {
        openDraft(item.draft);
        savedStatus.textContent = `Opened ${item.name}.`;
      }, `Open ${item.name}`);
      title.className = 'saved-name';
      const actions = document.createElement('div');
      actions.className = 'saved-actions';
      actions.append(
        button('Rename', () => showNameDialog(item), `Rename ${item.name}`),
        button('Delete', () => {
          savedDrafts = savedDrafts.filter(entry => entry.id !== item.id);
          saveDraftList(`Deleted ${item.name}.`);
          byElement('save-draft').focus();
        }, `Delete ${item.name}`)
      );
      row.append(title, actions);
      list.append(row);
      if (focusId === item.id) title.focus();
    }
  }

  function saveDraftList(message, focusId) {
    const stored = writeStorage(draftsKey, savedDrafts);
    renderSaved(focusId);
    savedStatus.textContent = stored ? message : `${message} Changes apply for this session only.`;
  }

  function showNameDialog(item) {
    editingDraftId = item?.id ?? null;
    byElement('dialog-title').textContent = item ? 'Rename draft' : 'Save draft';
    byElement('confirm-save').textContent = item ? 'Rename' : 'Save draft';
    draftName.value = item?.name ?? '';
    draftName.setCustomValidity('');
    nameDialog.showModal();
    draftName.select();
  }

  async function shareDraft() {
    const url = new URL(window.location.href);
    const params = new URLSearchParams(groups.map(group => [group, draft[group].join(',')]));
    url.search = '';
    url.hash = `draft?${params}`;
    try {
      await navigator.clipboard.writeText(url.href);
      savedStatus.textContent = 'Draft link copied.';
    } catch {
      const input = byElement('share-link');
      input.value = url.href;
      byElement('share-dialog').showModal();
      input.select();
    }
  }

  async function rankCarries() {
    const requestedRevision = revision;
    const payload = copyDraft(draft);
    if (poolFilter.checked) payload.candidate_pool = poolIds().filter(id => supported.has(id));
    pending = true;
    results.hidden = true;
    renderDraft();
    status.textContent = 'Ranking carries…';
    try {
      const response = window.recommendCarry ? await window.recommendCarry(payload) : await fetch('/api/recommend', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(payload)
      });
      const data = await response.json();
      if (revision !== requestedRevision) return;
      if (!response.ok) throw new Error(data.error || 'Recommendation failed.');
      const list = results.querySelector('ol');
      list.replaceChildren();
      for (const [index, item] of data.recommendations.entries()) {
        const row = document.createElement('li');
        const position = document.createElement('span');
        position.className = 'result-position';
        position.textContent = index + 1;
        const name = document.createElement('span');
        name.textContent = item.hero_name;
        const score = document.createElement('strong');
        score.textContent = item.model_score.toFixed(3);
        row.append(position, portrait(byId.get(item.hero_id)), name, score);
        list.append(row);
      }
      results.hidden = !data.recommendations.length;
      if (data.recommendations.length) status.textContent = `Showing ${data.recommendations.length} ranked carries.`;
      else status.textContent = poolFilter.checked
        ? 'No available heroes in your carry pool. Add favorites or turn off the filter.'
        : 'No available carries. Change picks or remove a ban.';
    } catch (error) {
      if (revision === requestedRevision) status.textContent = error.message;
    } finally {
      pending = false;
      renderDraft();
    }
  }

  buildTeams();
  buildRoster();
  byElement('search').addEventListener('input', updateSearch);
  for (const choice of ['pick', 'ban']) {
    byElement(`${choice}-mode`).addEventListener('click', () => {
      mode = choice;
      renderDraft();
    });
  }
  byElement('example').addEventListener('click', () => {
    openDraft({allies: [2, 5, 25, 86], enemies: [14, 26, 74, 99], banned: []});
  });
  byElement('clear').addEventListener('click', () => openDraft({allies: [], enemies: [], banned: []}));
  byElement('save-draft').addEventListener('click', () => showNameDialog());
  byElement('cancel-save').addEventListener('click', () => nameDialog.close());
  byElement('share-draft').addEventListener('click', shareDraft);
  draftName.addEventListener('input', () => draftName.setCustomValidity(''));
  byElement('draft-form').addEventListener('submit', event => {
    event.preventDefault();
    const name = draftName.value.trim();
    if (!name) {
      draftName.setCustomValidity('Enter a draft name.');
      draftName.reportValidity();
      return;
    }
    let id = editingDraftId;
    const renamed = id !== null;
    if (renamed) savedDrafts.find(item => item.id === id).name = name;
    else {
      id = globalThis.crypto.randomUUID?.() ?? `${Date.now()}-${Math.random()}`;
      savedDrafts.unshift({id, name, draft: copyDraft(draft)});
    }
    nameDialog.close();
    saveDraftList(renamed ? `Renamed to ${name}.` : `Saved ${name}.`, id);
  });
  recommend.addEventListener('click', rankCarries);
  poolFilter.checked = poolPreference ?? false;
  poolFilter.addEventListener('change', () => {
    poolPreference = poolFilter.checked;
    writeStorage(poolFilterKey, poolPreference);
    changeDraft();
  });
  window.addEventListener('lastpick:favorites', event => {
    if (poolFilter.checked) changeDraft();
    else renderDraft();
    if (!event.detail.stored) savedStatus.textContent = 'Carry pool changes apply for this session only.';
  });
  window.addEventListener('hashchange', showView);
  renderSaved();
  renderDraft();
  showView();
})();
