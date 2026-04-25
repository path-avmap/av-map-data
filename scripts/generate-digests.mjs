#!/usr/bin/env node
// Generate weekly AI digests of AV-Map events using Anthropic's Claude API.
//
// Usage:
//   node scripts/generate-digests.mjs            # Generate the most recent complete week
//   node scripts/generate-digests.mjs --backfill # Generate every week back to the dataset start
//   node scripts/generate-digests.mjs --force    # Re-generate even if already in digests.json
//
// Requires:
//   ANTHROPIC_API_KEY in env
//   @anthropic-ai/sdk + csv-parse installed

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { parse } from 'csv-parse/sync';
import Anthropic from '@anthropic-ai/sdk';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(__dirname, '..');
const EVENTS_CSV = path.join(REPO_ROOT, 'events.csv');
const DIGESTS_JSON = path.join(REPO_ROOT, 'digests.json');

const MODEL = 'claude-sonnet-4-5-20250929';

const args = new Set(process.argv.slice(2));
const BACKFILL = args.has('--backfill');
const FORCE = args.has('--force');

if (!process.env.ANTHROPIC_API_KEY) {
  console.error('ANTHROPIC_API_KEY env var is required');
  process.exit(1);
}

const client = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY });

// ── ISO week helpers ────────────────────────────────────────────────
// ISO week: weeks start on Monday; week 1 contains the year's first Thursday.

function toUtcMidnight(d) {
  return new Date(Date.UTC(d.getUTCFullYear(), d.getUTCMonth(), d.getUTCDate()));
}

function isoWeekStart(d) {
  const u = toUtcMidnight(d);
  const day = u.getUTCDay() || 7; // 1=Mon..7=Sun
  if (day !== 1) u.setUTCDate(u.getUTCDate() - (day - 1));
  return u;
}

function isoWeekKey(d) {
  // Returns "YYYY-Www" string per ISO 8601.
  const u = toUtcMidnight(d);
  u.setUTCDate(u.getUTCDate() + 4 - (u.getUTCDay() || 7)); // Thu of this week
  const yearStart = new Date(Date.UTC(u.getUTCFullYear(), 0, 1));
  const week = Math.ceil(((u - yearStart) / 86400000 + 1) / 7);
  return `${u.getUTCFullYear()}-W${String(week).padStart(2, '0')}`;
}

function addDays(d, n) {
  const x = new Date(d);
  x.setUTCDate(x.getUTCDate() + n);
  return x;
}

function fmtDate(d) {
  return d.toISOString().slice(0, 10);
}

// ── Load events ─────────────────────────────────────────────────────

function loadEvents() {
  const raw = fs.readFileSync(EVENTS_CSV, 'utf8');
  const rows = parse(raw, { columns: true, skip_empty_lines: true, trim: true });
  const events = rows
    .filter((r) => r.date && r.event_type && r.company)
    .map((r) => ({
      ...r,
      _date: new Date(`${r.date}T00:00:00Z`),
    }))
    .filter((r) => !isNaN(r._date.getTime()));
  events.sort((a, b) => a._date - b._date);
  return events;
}

function groupByWeek(events) {
  const weeks = new Map();
  for (const e of events) {
    const key = isoWeekKey(e._date);
    if (!weeks.has(key)) {
      const start = isoWeekStart(e._date);
      weeks.set(key, { week: key, start, end: addDays(start, 6), events: [] });
    }
    weeks.get(key).events.push(e);
  }
  return [...weeks.values()].sort((a, b) => a.start - b.start);
}

// ── Prompt construction ─────────────────────────────────────────────

const SYSTEM_PROMPT = `You are an analyst writing concise weekly briefings for the AV Map (avmap.io), which tracks the global rollout of autonomous vehicle (AV) services.

You will receive a list of events that occurred during a single week. Each event describes a real-world change to an AV deployment (a launch, a policy change, an area expansion, etc.).

Write ONE plain-prose paragraph (3-5 sentences, ~80 words) capturing the most important narrative threads of the week. Lead with what changed strategically, not a list. Mention specific companies and cities. Be terse. No hedging. No "in conclusion." Plain English.

Also produce a 6-12 word headline that captures the dominant story.

Output STRICT JSON only, no prose around it:
{ "headline": "...", "summary": "..." }

If the week has only one or two minor events, say so briefly — do not pad.`;

function eventLine(e) {
  const parts = [
    e.date,
    e.event_type,
    e.company,
    e.city || '',
    e.access || '',
    e.supervision || '',
    e.fares ? `fares=${e.fares}` : '',
    e.notes ? `(${e.notes})` : '',
  ].filter(Boolean);
  return parts.join(' | ');
}

async function generateDigest(weekBucket) {
  const eventsBlock = weekBucket.events.map(eventLine).join('\n');
  const userPrompt = `Week: ${fmtDate(weekBucket.start)} to ${fmtDate(weekBucket.end)}\nEvents:\n${eventsBlock}`;

  const resp = await client.messages.create({
    model: MODEL,
    max_tokens: 400,
    system: [
      { type: 'text', text: SYSTEM_PROMPT, cache_control: { type: 'ephemeral' } },
    ],
    messages: [{ role: 'user', content: userPrompt }],
  });

  const text = resp.content.map((b) => (b.type === 'text' ? b.text : '')).join('').trim();
  // Strip code fences if present.
  const cleaned = text.replace(/^```json\s*/i, '').replace(/^```\s*/, '').replace(/```\s*$/, '').trim();

  let parsed;
  try {
    parsed = JSON.parse(cleaned);
  } catch (err) {
    throw new Error(`Failed to parse model JSON: ${cleaned.slice(0, 200)}`);
  }
  if (typeof parsed.summary !== 'string' || typeof parsed.headline !== 'string') {
    throw new Error(`Model returned unexpected shape: ${JSON.stringify(parsed).slice(0, 200)}`);
  }
  return parsed;
}

// ── Main ────────────────────────────────────────────────────────────

function loadExistingDigests() {
  if (!fs.existsSync(DIGESTS_JSON)) return [];
  try {
    return JSON.parse(fs.readFileSync(DIGESTS_JSON, 'utf8'));
  } catch {
    return [];
  }
}

function saveDigests(digests) {
  digests.sort((a, b) => (a.week < b.week ? 1 : -1)); // newest first
  fs.writeFileSync(DIGESTS_JSON, JSON.stringify(digests, null, 2) + '\n');
}

async function main() {
  const events = loadEvents();
  const weeks = groupByWeek(events);

  // Exclude the current (incomplete) week — only generate completed weeks.
  const today = toUtcMidnight(new Date());
  const currentWeekStart = isoWeekStart(today);
  const completed = weeks.filter((w) => w.start < currentWeekStart);

  const targets = BACKFILL ? completed : completed.slice(-1);

  const existing = loadExistingDigests();
  const existingByWeek = new Map(existing.map((d) => [d.week, d]));

  console.log(`Total weeks with events: ${completed.length}`);
  console.log(`Targets this run: ${targets.length} (${BACKFILL ? 'backfill' : 'latest only'})`);
  console.log(`Already cached: ${existing.length}`);

  let generated = 0;
  let skipped = 0;
  let failed = 0;

  for (const w of targets) {
    if (!FORCE && existingByWeek.has(w.week)) {
      skipped++;
      continue;
    }
    try {
      console.log(`→ ${w.week}  (${w.events.length} events)`);
      const { headline, summary } = await generateDigest(w);
      const entry = {
        week: w.week,
        start: fmtDate(w.start),
        end: fmtDate(w.end),
        headline,
        summary,
        event_count: w.events.length,
        event_types: [...new Set(w.events.map((e) => e.event_type))].sort(),
        companies: [...new Set(w.events.map((e) => e.company))].sort(),
        generated_at: new Date().toISOString(),
      };
      existingByWeek.set(w.week, entry);
      generated++;
      // Persist incrementally so a crash mid-backfill doesn't lose progress.
      saveDigests([...existingByWeek.values()]);
    } catch (err) {
      console.error(`  ✗ ${w.week} failed: ${err.message}`);
      failed++;
    }
  }

  console.log(`\nDone. generated=${generated} skipped=${skipped} failed=${failed}`);
  if (failed > 0) process.exit(1);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
