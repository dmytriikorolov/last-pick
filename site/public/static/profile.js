(() => {
  'use strict';

  const DAY = 86_400_000;
  const MATCH_LIMIT = 500;
  const AUTO_POOL_LIMIT = 8;
  const CACHE_AGE = 30 * 60 * 1000;
  const ACCOUNT_MAX = 4_294_967_295n;
  const STEAM_ID_BASE = 76_561_197_960_265_728n;
  const PROFILE_KEY = 'lastpick.profile.v1';
  const shortDate = new Intl.DateTimeFormat('en', { month: 'short', day: 'numeric', timeZone: 'UTC' });
  const longDate = new Intl.DateTimeFormat('en', { dateStyle: 'long', timeZone: 'UTC' });
  const monthName = new Intl.DateTimeFormat('en', { month: 'short', timeZone: 'UTC' });
  const matchCount = count => `${count} ${count === 1 ? 'match' : 'matches'}`;

  function parseAccount(input) {
    let value = input.trim();
    if (!/^\d{1,17}$/.test(value)) {
      let url;
      try {
        url = new URL(value);
      } catch {
        return null;
      }
      const path = url.hostname === 'steamcommunity.com'
        ? url.pathname.match(/^\/profiles\/(\d{17})\/?$/)
        : /^(?:www\.)?opendota\.com$/.test(url.hostname)
          ? url.pathname.match(/^\/players\/(\d{1,10})\/?$/) : null;
      if (!path || url.protocol !== 'https:'
          || url.username || url.password || url.port) return null;
      value = path[1];
    }
    const id = BigInt(value);
    const account = value.length === 17 ? id - STEAM_ID_BASE : id;
    return account > 0n && account <= ACCOUNT_MAX ? String(account) : null;
  }

  function calendarRange(now = Date.now()) {
    const today = new Date(now);
    const end = Date.UTC(today.getUTCFullYear(), today.getUTCMonth(), today.getUTCDate());
    const first = new Date(Date.UTC(today.getUTCFullYear(), today.getUTCMonth() - 6, 1));
    const lastDay = new Date(Date.UTC(first.getUTCFullYear(), first.getUTCMonth() + 1, 0)).getUTCDate();
    first.setUTCDate(Math.min(today.getUTCDate(), lastDay));
    return { start: first.getTime(), end };
  }

  function matchRows(rows) {
    const unique = new Map();
    for (const value of rows.slice(0, MATCH_LIMIT)) {
      if (!value || !/^\d+$/.test(String(value.match_id)) || value.match_id === 0
          || !Number.isInteger(value.start_time) || value.start_time <= 0
          || !Number.isInteger(value.hero_id) || value.hero_id <= 0) continue;
      const id = String(value.match_id);
      const slot = value.player_slot;
      const radiant = Number.isInteger(slot) && slot >= 0 && slot <= 4;
      const dire = Number.isInteger(slot) && slot >= 128 && slot <= 132;
      unique.set(id, {
        id,
        date: value.start_time * 1000,
        hero: value.hero_id,
        win: (radiant || dire) && typeof value.radiant_win === 'boolean'
          ? radiant === value.radiant_win : null,
        minutes: Number.isFinite(value.duration) && value.duration > 0
          ? Math.floor(value.duration / 60) : null,
      });
    }
    return [...unique.values()].sort((a, b) => b.date - a.date);
  }

  function summary(matches) {
    const players = new Map();
    let wins = 0;
    let finished = 0;
    for (const match of matches) {
      const count = players.get(match.hero) || { hero: match.hero, games: 0, wins: 0, results: 0 };
      count.games++;
      if (match.win !== null) {
        count.results++;
        finished++;
        if (match.win) {
          count.wins++;
          wins++;
        }
      }
      players.set(match.hero, count);
    }
    const heroes = [...players.values()].sort((a, b) => b.games - a.games || a.hero - b.hero);
    return { heroes, wins, finished };
  }

  function automaticPool(matches, supported) {
    const played = new Map();
    for (const { hero, date } of matches) {
      if (!supported.has(hero)) continue;
      const entry = played.get(hero) || { hero, games: 0, latest: date };
      entry.games++;
      entry.latest = Math.max(entry.latest, date);
      played.set(hero, entry);
    }
    return [...played.values()]
      .sort((a, b) => b.games - a.games || b.latest - a.latest || a.hero - b.hero)
      .slice(0, AUTO_POOL_LIMIT);
  }

  function exampleRows(end) {
    let seed = 8731;
    function random() {
      seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
      return seed / 4294967296;
    }
    const choices = [8, 8, 8, 11, 11, 6, 6, 93, 1, 10, 94];
    const rows = [];
    for (let age = 0; age <= 184; age++) {
      const weekday = new Date(end - age * DAY).getUTCDay();
      const weekend = weekday === 0 || weekday === 6;
      if (age > 2 && random() > (weekend ? 0.72 : 0.42)) continue;
      const games = 1 + Math.floor(random() * (weekend ? 5 : 3));
      for (let game = 0; game < games; game++) {
        const time = end - age * DAY + (game + 8) * 3_600_000;
        rows.push({
          id: String(rows.length + 1),
          date: Math.min(time, Date.now()),
          hero: choices[Math.floor(random() * choices.length)],
          win: random() < 0.57,
          minutes: 24 + Math.floor(random() * 28),
        });
      }
    }
    return rows.sort((a, b) => b.date - a.date);
  }

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = { parseAccount, calendarRange, exampleRows, matchRows, summary, automaticPool };
  }
  if (typeof document === 'undefined') return;

  const view = document.getElementById('profile-view');
  if (!view) return;
  const heroes = new Map(JSON.parse(document.getElementById('hero-data').textContent)
    .map(hero => [hero.id, hero]));
  const supported = new Set(JSON.parse(document.getElementById('candidate-data').textContent));
  const favorites = window.lastpickFavorites;
  let current;
  let historyPool = [];
  let daySelected;
  let request;
  let restore;
  let matchOffset = 0;

  view.innerHTML = `
    <header class="profile-top">
      <div>
        <h1>My Dota</h1>
        <p>See when you play and build a carry pool for your next draft.</p>
      </div>
      <form class="profile-form">
        <label for="profile-account">OpenDota ID or profile link</label>
        <div class="profile-form-actions">
          <input id="profile-account" name="account" type="text" maxlength="160" placeholder="e.g. 86745912" spellcheck="false" autocomplete="off" required>
          <button type="submit" class="profile-primary">Load profile</button>
        </div>
      </form>
    </header>
    <p class="profile-status" role="status" aria-live="polite"></p>
    <div class="profile-identity">
      <div class="profile-identity-detail">
        <span class="profile-identity-mark" aria-hidden="true"></span>
        <div><strong class="profile-name"></strong><p class="profile-source"></p></div>
      </div>
      <button type="button" class="profile-example">View example</button>
    </div>
    <section class="profile-card profile-activity" aria-labelledby="profile-activity-title">
      <div class="profile-card-head">
        <div><h2 id="profile-activity-title">Match activity</h2><p>Past six months · UTC</p></div>
        <div class="profile-stats"></div>
      </div>
      <div class="profile-calendar-scroll">
        <div class="profile-calendar" role="group" aria-label="Match activity by day" aria-describedby="profile-calendar-help"></div>
      </div>
      <span id="profile-calendar-help" class="profile-sr-only">Use arrow keys to move between days. Press Enter or Space to filter the match list.</span>
      <div class="profile-legend" aria-label="Lighter squares indicate more matches">
        <span>Fewer</span><i data-level="0"></i><i data-level="1"></i><i data-level="2"></i><i data-level="3"></i><i data-level="4"></i><span>More</span>
      </div>
    </section>
    <div class="profile-columns">
      <section class="profile-card" aria-labelledby="profile-matches-title">
        <div class="profile-card-head">
          <div><h2 id="profile-matches-title">Recent matches</h2><p class="profile-match-subtitle"></p></div>
          <button type="button" class="profile-clear-date" hidden>Show all days</button>
        </div>
        <ul class="profile-match-list"></ul>
        <button type="button" class="profile-more" hidden>Show more matches</button>
      </section>
      <section class="profile-card profile-heroes" aria-labelledby="profile-heroes-title">
        <div class="profile-card-head">
          <div><h2 id="profile-heroes-title">Most played</h2><p>Past six months</p></div>
        </div>
        <ul class="profile-hero-list"></ul>
      </section>
    </div>
    <section class="profile-card profile-pool" aria-labelledby="profile-pool-title">
      <div class="profile-card-head">
        <div><h2 id="profile-pool-title">My carry pool</h2><p>Up to eight supported heroes from your match history, plus your favorites.</p></div>
        <a href="#draft">Open draft</a>
      </div>
      <ul class="profile-pool-list"></ul>
    </section>`;

  const form = view.querySelector('.profile-form');
  const input = view.querySelector('#profile-account');
  const status = view.querySelector('.profile-status');
  const calendar = view.querySelector('.profile-calendar');
  const calendarScroll = view.querySelector('.profile-calendar-scroll');
  const submit = form.querySelector('button[type="submit"]');

  function element(tag, className, text = '') {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text) node.textContent = text;
    return node;
  }

  function favoriteButton(hero) {
    const saved = favorites.has(hero.id);
    const button = element('button', 'profile-favorite', saved ? '★' : '☆');
    button.type = 'button';
    button.setAttribute('aria-pressed', String(saved));
    button.setAttribute('aria-label', `Favorite ${hero.name}`);
    button.dataset.favorite = String(hero.id);
    button.onclick = () => favorites.toggle(hero.id);
    return button;
  }

  function portrait(hero) {
    if (!hero) return element('span', 'profile-portrait profile-unknown-hero', '?');
    const image = element('img', 'profile-portrait');
    image.src = hero.image;
    image.alt = '';
    image.loading = 'lazy';
    return image;
  }

  function note(node, text) {
    node.replaceChildren(element('li', 'profile-empty', text));
  }

  function renderLists() {
    const list = view.querySelector('.profile-match-list');
    const subtitle = view.querySelector('.profile-match-subtitle');
    const showAll = view.querySelector('.profile-clear-date');
    const more = view.querySelector('.profile-more');
    const selected = daySelected === undefined
      ? current.matches
      : current.matches.filter(match => Math.floor(match.date / DAY) * DAY === daySelected);
    showAll.hidden = daySelected === undefined;
    more.hidden = selected.length <= matchOffset + 8;
    if (daySelected === undefined) {
      subtitle.textContent = `${matchCount(selected.length)} in available history`;
    } else {
      const count = daySelected < current.coverageStart ? 'History unavailable' : matchCount(selected.length);
      subtitle.textContent = `${longDate.format(daySelected)} · ${count}`;
    }
    list.replaceChildren();
    if (selected.length === 0) {
      note(list, 'No matches available for this period.');
      return;
    }

    for (const match of selected.slice(0, matchOffset + 8)) {
      const hero = heroes.get(match.hero);
      const row = element('li', 'profile-list-row');
      const details = element('div', 'profile-row-detail');
      const name = hero?.name || `Hero ${match.hero}`;
      if (current.example) {
        details.append(element('strong', null, name));
      } else {
        const link = element('a', null, name);
        link.href = `https://www.opendota.com/matches/${match.id}`;
        link.setAttribute('aria-label', `${name} · Open match on OpenDota`);
        link.target = '_blank';
        link.rel = 'noopener noreferrer';
        details.append(link);
      }
      details.append(element('span', null, `${shortDate.format(match.date)}${match.minutes === null ? '' : ` · ${match.minutes} min`}`));
      const result = match.win === null ? 'Unknown' : match.win ? 'Win' : 'Loss';
      row.append(portrait(hero), details, element('span', `profile-result profile-${result.toLowerCase()}`, result));
      list.append(row);
    }
  }

  function renderFavorites() {
    const mostPlayed = view.querySelector('.profile-hero-list');
    mostPlayed.replaceChildren();
    for (const stat of summary(current.matches).heroes.slice(0, 5)) {
      const hero = heroes.get(stat.hero);
      const row = element('li', 'profile-list-row');
      const details = element('div', 'profile-row-detail');
      const rate = stat.results ? ` · ${Math.round(100 * stat.wins / stat.results)}% wins` : '';
      details.append(
        element('strong', null, hero?.name || `Hero ${stat.hero}`),
        element('span', null, `${matchCount(stat.games)}${rate}`)
      );
      row.append(portrait(hero), details);
      if (hero) row.append(favoriteButton(hero));
      mostPlayed.append(row);
    }
    if (!mostPlayed.children.length) note(mostPlayed, 'No heroes in this period.');

    const pool = view.querySelector('.profile-pool-list');
    pool.replaceChildren();
    for (const id of favorites.poolIds()) {
      const hero = heroes.get(id);
      if (!hero) continue;
      const row = element('li', 'profile-list-row');
      const details = element('div', 'profile-row-detail');
      const fromHistory = historyPool.find(entry => entry.hero === id);
      const description = fromHistory ? `Auto-selected · ${matchCount(fromHistory.games)} played`
        : supported.has(id) ? 'Favorite · Can be recommended' : 'Outside the model’s carry pool';
      details.append(
        element('strong', null, hero.name),
        element('span', null, description)
      );
      row.append(portrait(hero), details, favoriteButton(hero));
      pool.append(row);
    }
    if (!pool.children.length) note(pool, current.example
      ? 'Load your profile to build a pool automatically, or star heroes to add them yourself.'
      : 'No supported carries found in this history. Star heroes to add your favorites.');
  }

  function selectDay(day) {
    daySelected = day;
    matchOffset = 0;
    for (const button of calendar.querySelectorAll('button[data-day]')) {
      button.setAttribute('aria-pressed', String(Number(button.dataset.day) === daySelected));
    }
    renderLists();
  }

  function renderCalendar(start, end) {
    calendar.replaceChildren();
    const counts = new Map();
    for (const match of current.matches) {
      const date = Math.floor(match.date / DAY) * DAY;
      counts.set(date, (counts.get(date) || 0) + 1);
    }
    const startOffset = (new Date(start).getUTCDay() + 6) % 7;
    const columns = Math.floor(((end - start) / DAY + startOffset) / 7) + 1;
    calendar.style.setProperty('--profile-weeks', columns);
    for (const [index, name] of [[2, 'Mon'], [4, 'Wed'], [6, 'Fri']]) {
      const weekday = element('span', 'profile-weekday', name);
      weekday.style.gridRow = index;
      weekday.style.gridColumn = '1';
      weekday.setAttribute('aria-hidden', 'true');
      calendar.append(weekday);
    }
    let lastLabel = -4;
    for (let day = start; day <= end; day += DAY) {
      const index = (day - start) / DAY + startOffset;
      const column = Math.floor(index / 7) + 2;
      if (day === start || (new Date(day).getUTCDate() === 1 && column > lastLabel + 1)) {
        const month = element('span', 'profile-month', monthName.format(day));
        month.style.gridColumn = column;
        month.style.gridRow = '1';
        month.setAttribute('aria-hidden', 'true');
        calendar.append(month);
        lastLabel = column;
      }
      const count = counts.get(day) || 0;
      const cell = element('button', 'profile-day');
      const missing = day < current.coverageStart;
      const label = `${longDate.format(day)} · ${missing ? 'History unavailable' : matchCount(count)}`;
      cell.type = 'button';
      cell.title = label;
      cell.dataset.level = missing ? 'unknown' : Math.min(count, 4);
      cell.dataset.day = day;
      cell.style.gridColumn = column;
      cell.style.gridRow = index % 7 + 2;
      cell.tabIndex = day === end ? 0 : -1;
      cell.setAttribute('aria-pressed', 'false');
      cell.setAttribute('aria-label', label);
      cell.onclick = () => {
        calendar.querySelector('button[tabindex="0"]').tabIndex = -1;
        cell.tabIndex = 0;
        selectDay(day === daySelected ? undefined : day);
      };
      calendar.append(cell);
    }
  }

  calendar.addEventListener('keydown', event => {
    const offset = { ArrowUp: -1, ArrowDown: 1, ArrowLeft: -7, ArrowRight: 7 }[event.key];
    if (!offset || event.target.dataset.day === undefined) return;
    event.preventDefault();
    const next = Number(event.target.dataset.day) + offset * DAY;
    const target = calendar.querySelector(`button[data-day="${next}"]`);
    if (!target) return;
    event.target.tabIndex = -1;
    target.tabIndex = 0;
    target.focus();
  });

  function showProfile(profile) {
    const { start, end } = calendarRange();
    current = {
      ...profile,
      matches: profile.matches.filter(match => match.date >= start && match.date < end + DAY),
      coverageStart: profile.capped && profile.matches.length
        ? Math.floor(profile.matches.at(-1).date / DAY) * DAY : start,
    };
    daySelected = undefined;
    matchOffset = 0;
    view.querySelector('.profile-name').textContent = profile.example ? 'Example profile' : `Dota account ${profile.account}`;
    view.querySelector('.profile-source').textContent = profile.example ? 'Sample matches for exploring the profile' : 'Available match history from OpenDota';
    view.querySelector('.profile-example').hidden = profile.example;
    const stat = summary(current.matches);
    const stats = view.querySelector('.profile-stats');
    const activeDays = new Set(current.matches.map(match => Math.floor(match.date / DAY))).size;
    stats.replaceChildren();
    const values = [
      [current.matches.length, 'Matches shown'],
      [activeDays, 'Active days'],
      [stat.finished ? `${Math.round(stat.wins / stat.finished * 100)}%` : '—', 'Win rate'],
    ];
    for (const [value, label] of values) {
      const metric = element('div', 'profile-stat');
      metric.append(element('strong', null, String(value)), element('span', null, label));
      stats.append(metric);
    }
    renderCalendar(start, end);
    showLatestDay();
    renderLists();
    renderFavorites();
  }

  function showLatestDay() {
    requestAnimationFrame(() => {
      if (!view.hidden) calendarScroll.scrollLeft = calendarScroll.scrollWidth;
    });
  }

  function storageRead() {
    try {
      const saved = JSON.parse(localStorage.getItem(PROFILE_KEY));
      if (saved && parseAccount(saved.account || '') === saved.account && Array.isArray(saved.rows)) {
        return saved;
      }
    } catch { /* The profile works when browser storage is unavailable. */ }
    return null;
  }

  function storageWrite(value) {
    try {
      localStorage.setItem(PROFILE_KEY, JSON.stringify(value));
    } catch { /* The profile works when browser storage is unavailable. */ }
  }

  function showExample() {
    request?.abort();
    request = undefined;
    submit.disabled = false;
    status.textContent = '';
    showProfile({ example: true, matches: exampleRows(calendarRange().end) });
  }

  function displayRows(account, rows, cached) {
    const matches = matchRows(rows);
    const { start, end } = calendarRange();
    historyPool = automaticPool(matches.filter(match => match.date >= start && match.date < end + DAY), supported);
    favorites.setHistoryPool(historyPool.map(entry => entry.hero));
    showProfile({ account, matches, capped: rows.length >= MATCH_LIMIT });
    if (current.matches.length === 0) {
      status.textContent = 'OpenDota returned no matches for this period. Check the ID or enable Expose Public Match Data in Dota.';
    } else if (rows.length >= MATCH_LIMIT) {
      status.textContent = `Showing up to ${MATCH_LIMIT} matches. Earlier days can have incomplete coverage.`;
    } else {
      status.textContent = cached ? 'Showing match history saved on this device.' : '';
    }
  }

  form.onsubmit = async event => {
    event.preventDefault();
    const account = parseAccount(input.value);
    if (!account) {
      status.textContent = 'Enter an OpenDota account ID, an OpenDota player link, or a Steam link with a numeric profile ID.';
      input.focus();
      return;
    }
    request?.abort();
    request = undefined;
    const cached = storageRead();
    if (cached?.account === account && Date.now() - cached.at < CACHE_AGE) {
      submit.disabled = false;
      displayRows(account, cached.rows, true);
      return;
    }
    const controller = new AbortController();
    request = controller;
    submit.disabled = true;
    let timeout = false;
    const timer = setTimeout(() => {
      timeout = true;
      controller.abort();
    }, 9000);
    status.textContent = 'Loading match history…';
    try {
      // OpenDota's default history backend can time out on public profiles.
      const params = new URLSearchParams({ limit: String(MATCH_LIMIT), date: '185', sort: 'start_time', clickhouse: '1' });
      const response = await fetch(`https://api.opendota.com/api/players/${account}/matches?${params}`, {
        signal: controller.signal,
        credentials: 'omit',
        referrerPolicy: 'no-referrer',
      });
      if (response.status === 429) throw new Error('OpenDota is limiting requests. Try again in a few minutes.');
      if (response.status === 404) throw new Error('OpenDota could not find this account. Check the ID and try again.');
      if (!response.ok) throw new Error('OpenDota could not load this match history. Try again shortly.');
      const rows = await response.json();
      if (!Array.isArray(rows)) throw new Error('OpenDota returned an unexpected response. Try again shortly.');
      if (request !== controller) return;
      displayRows(account, rows, false);
      storageWrite({ account, rows: rows.slice(0, MATCH_LIMIT), at: Date.now() });
    } catch (error) {
      if (request !== controller) return;
      const message = timeout ? 'OpenDota took too long. Try again shortly.'
        : error instanceof TypeError || error instanceof SyntaxError
          ? 'Could not connect to OpenDota. Try again shortly.' : error.message;
      if (cached?.account === account) {
        displayRows(account, cached.rows, true);
        status.textContent = `${message} Showing match history saved on this device.`;
      } else {
        status.textContent = message;
      }
    } finally {
      clearTimeout(timer);
      if (request === controller) {
        request = undefined;
        submit.disabled = false;
      }
    }
  };

  view.querySelector('.profile-example').onclick = showExample;
  view.querySelector('.profile-clear-date').onclick = () => selectDay(undefined);
  view.querySelector('.profile-more').onclick = () => {
    matchOffset += 8;
    renderLists();
  };
  window.addEventListener('hashchange', () => {
    if (location.hash === '#profile') showLatestDay();
  });
  window.addEventListener('lastpick:favorites', () => {
    if (!current) return;
    const focused = document.activeElement;
    if (focused?.dataset.favorite && view.contains(focused)) {
      restore = [focused.closest('ul').className, focused.dataset.favorite];
    }
    renderFavorites();
    if (restore) {
      const replacement = view.querySelector(`.${restore[0]} button[data-favorite="${restore[1]}"]`);
      const next = replacement || view.querySelector('.profile-pool-list button, .profile-pool a');
      next?.focus();
      restore = undefined;
    }
  });

  const cached = storageRead();
  if (cached) {
    input.value = cached.account;
    displayRows(cached.account, cached.rows, true);
  } else {
    showExample();
  }
})();
