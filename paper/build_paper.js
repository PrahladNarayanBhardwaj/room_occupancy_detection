// Builds the manuscript (paper/Room_Occupancy_Paper.docx) from analysis/reanalysis_results.json.
// Run the notebook first, then: npm install docx && node paper/build_paper.js
const fs = require('fs');
const path = require('path');
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, HeadingLevel, Header,
  AlignmentType, WidthType, ShadingType, BorderStyle, LevelFormat, PageNumber, PageBreak, LineNumberRestartFormat, LineRuleType,
} = require('docx');

const R = JSON.parse(fs.readFileSync(path.join(__dirname, '../analysis/reanalysis_results.json')));
const FIG = path.join(__dirname, 'figures');
const FONT = 'Times New Roman';

// Details still to fill in (shown highlighted in the document)
const AFFILIATION = 'Independent Researcher';
const WITH_PHONE = process.argv.includes('--submission');   // phone number only in the journal-submission version
const PHONE = '(848) 262-3770';
const ORCID = 'https://orcid.org/0009-0008-7598-1954';
const EMAIL = 'prahlad.narayan77@gmail.com';
const CODE_URL = 'https://github.com/PrahladNarayanBhardwaj/room-occupancy-detection';

// ---------- helpers ----------
let counting = null;                         // 'abstract' | 'body' | null
const words = { abstract: 0, body: 0 };
const count = (t) => { if (counting) words[counting] += t.replace(/\*/g, '').split(/\s+/).filter(w => /[A-Za-z0-9]/.test(w)).length; };

function runs(text, base = {}) {             // supports **bold**, *italic* and [ADD: ...] placeholders
  const out = [];
  const re = /(\*\*[^*]+\*\*|\*[^*]+\*|\[ADD:[^\]]+\])/g;
  let last = 0, m;
  while ((m = re.exec(text))) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), ...base }));
    const t = m[0];
    if (t.startsWith('**')) out.push(new TextRun({ text: t.slice(2, -2), bold: true, ...base }));
    else if (t.startsWith('[ADD')) out.push(new TextRun({ text: t, highlight: 'yellow', ...base }));
    else out.push(new TextRun({ text: t.slice(1, -1), italics: true, ...base }));
    last = m.index + t.length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), ...base }));
  return out;
}
const DOUBLE = { line: 360, before: 0, after: 0 };
const P = (text, opts = {}) => { count(text); const p = new Paragraph({ children: runs(text), spacing: DOUBLE, indent: { firstLine: 720 }, ...opts }); p._text = text; return p; };
const PN = (text) => P(text, { indent: { firstLine: 0 } });     // paragraph without first-line indent
const H1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(t)] });
const H2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(t)] });
const LIST = (ref) => (t) => { count(t); return new Paragraph({ numbering: { reference: ref, level: 0 }, children: runs(t), spacing: DOUBLE }); };
const BUL = LIST('bullets');
const brk = () => new Paragraph({ children: [new PageBreak()] });

const f3 = (x) => Number(x).toFixed(3);
const pct = (x, d = 1) => (100 * Number(x)).toFixed(d) + '%';

// Tables (title above) and figures (caption below) are collected and placed at the end of the manuscript.
const tables = [], figures = [];
const border = { style: BorderStyle.SINGLE, size: 4, color: '808080' };
function table(title, header, rows, widths, note) {
  tables.push({ title, header, rows, widths, note });
  return `Table ${tables.length}`;
}
function figure(file, caption, width = 600) { figures.push({ file, caption, width }); return `Figure ${figures.length}`; }
function renderTable(t, i) {
  const total = t.widths.reduce((a, b) => a + b, 0);
  const cell = (s, j, head, last = false) => new TableCell({
    width: { size: t.widths[j], type: WidthType.DXA },
    borders: { top: border, bottom: border, left: border, right: border },
    shading: head ? { fill: 'EDEDED', type: ShadingType.CLEAR, color: 'auto' } : undefined,
    margins: { top: 40, bottom: 40, left: 80, right: 80 },
    children: [new Paragraph({ keepNext: !last, keepLines: true, spacing: { line: 240 }, alignment: j === 0 ? AlignmentType.LEFT : AlignmentType.CENTER, children: runs(String(s), { size: 20, bold: head || undefined }) })],
  });
  const out = [
    new Paragraph({ spacing: { before: 240, after: 120, line: 276 }, keepNext: true, children: runs(`**Table ${i + 1}.** ${t.title}`, { size: 22 }) }),
    new Table({ width: { size: total, type: WidthType.DXA }, columnWidths: t.widths,
      rows: [new TableRow({ tableHeader: true, children: t.header.map((h, j) => cell(h, j, true)) }),
             ...t.rows.map((r, k) => new TableRow({ cantSplit: true, children: r.map((c, j) => cell(c, j, false, k === t.rows.length - 1)) }))] }),
  ];
  out.push(new Paragraph({ spacing: { after: 240 }, children: [] }));
  return out;
}
function pngSize(f) { const b = fs.readFileSync(f); return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) }; }
function renderFigure(fg, i) {
  const f = path.join(FIG, fg.file); const s = pngSize(f); const w = fg.width;
  const h = Math.round(w * s.h / s.w); const scale = Math.min(1, 700 / h);
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, spacing: { line: 240, lineRule: LineRuleType.AUTO, before: 0, after: 0 }, children: [new ImageRun({ type: 'png', data: fs.readFileSync(f), transformation: { width: Math.round(w * scale), height: Math.round(h * scale) } })] }),
    new Paragraph({ spacing: { before: 120, after: 240, line: 276 }, children: runs(`**Figure ${i + 1}.** ${fg.caption}`, { size: 22 }) }),
  ];
}

// ---------- numbers ----------
const L = Object.fromEntries(R.lodo.map(r => [r.model, r]));
const rep = Object.fromEntries(R.reproduction.map(r => [r.model, r]));
const pol = (m, N) => R.policy.find(r => r.model === m && r.N === N);
const still = (m, s) => R.stillness.find(r => r.model === m && r.silence_min === s);
const abl = Object.fromEntries(R.ablation.map(r => [r.removed, r]));
const hc = R.headcount;
const hcRandom = hc.find(r => r.evaluation === 'random 75/25 split');
const hcBest = hc.find(r => r.features === 'with light' && r.model === 'Random forest' && r.evaluation === 'leave-one-day-out');
const LRo = L['LR original (unscaled)'], RFo = L['RF original (1000 trees, unpruned)'], GB = L['LightGBM, extended features'];
const RULE = L['Rule: motion + (CO2 or sound)'], PIR5 = L['Rule: motion in last 5 min'];
const GBname = 'LightGBM, extended features', RULEname = 'Rule: motion + (CO2 or sound)', LRname = 'LR original (unscaled)';

// ---------- tables and figures (numbered in order of first mention) ----------
const T1 = table('Snapshots per day in the dataset. Only three of the seven days have anyone in the room; the second column is the day of the week.',
  ['Date', 'Day', 'Snapshots', 'Occupied snapshots', 'Role in original split'],
  [['2017-12-22', 'Fri', '1,462', '825', 'train'], ['2017-12-23', 'Sat', '2,779', '782', 'train'], ['2017-12-24', 'Sun', '1,064', '0', 'train'],
   ['2017-12-25', 'Mon', '1,716', '0', 'train'], ['2017-12-26', 'Tue', '1,063', '0', 'train'], ['2018-01-10', 'Wed', '997', '294', 'test'], ['2018-01-11', 'Thu', '1,048', '0', 'test']],
  [1700, 900, 1500, 2000, 3260]);
const T2 = table('Results of the original analysis (train 22–26 December, test 10–11 January; 294 occupied and 1,751 empty test snapshots). TN/FP/FN/TP = true negatives, false positives (false alarms), false negatives (misses), true positives.',
  ['Model', 'TN', 'FP', 'FN', 'TP', 'Recall'],
  ['LR baseline', 'LR balanced', 'LR balanced + time_since_motion', 'LR no light (final model)', 'RF 1000 trees no light', 'RF WITH light (not tried originally)']
    .map(k => [k.replace('LR', 'Logistic regression').replace('RF', 'Random forest').replace(' (not tried originally)', ' (new check)'), rep[k].TN, rep[k].FP, rep[k].FN, rep[k].TP, f3(rep[k].recall)]),
  [4200, 1000, 1000, 1000, 1000, 1160]);
const T3names = [
  ['LR original (unscaled)', 'Logistic regression, original features (final model)'], ['LR original + light', 'Logistic regression, original features + light'],
  ['RF original (1000 trees, unpruned)', 'Random forest, original features'], ['RF original + light', 'Random forest, original features + light'],
  ['Rule: motion in last 5 min', 'Rule: motion in last 5 min'], ['Rule: motion + (CO2 or sound)', 'Rule: motion + (CO₂ rise or sound)'],
  ['LR scaled', 'Logistic regression, standardised'], ['LR scaled, extended features', 'Logistic regression, extended features'],
  ['RF regularised (min leaf 20)', 'Random forest, min. leaf 20'], ['RF regularised, extended features', 'Random forest, min. leaf 20, extended'],
  ['LightGBM', 'LightGBM, original features'], ['LightGBM, extended features', 'LightGBM, extended features'],
  ['SVM (RBF, scaled)', 'SVM (RBF kernel)'], ['kNN (k=15, scaled)', 'k-nearest neighbours (k = 15)'], ['LDA', 'Linear discriminant analysis'], ['Decision tree (depth 3)', 'Decision tree (depth 3)']];
const T3 = table('Leave-one-day-out results for occupied-versus-empty detection. Each of the seven days was held out in turn; predictions from all seven folds were pooled (1,901 occupied and 8,228 empty snapshots). The first four rows are the models of the original analysis. False-alarm rate = false positives / empty snapshots. PR-AUC = area under the precision–recall curve.',
  ['Model', 'Recall', 'Precision', 'F1', 'False-alarm rate', 'False alarms on 25 Dec', 'PR-AUC'],
  T3names.map(([k, label]) => [label, f3(L[k].recall), f3(L[k].precision), f3(L[k].F1), pct(L[k].FP_rate), L[k].FP_Dec25, f3(L[k].PR_AUC)]),
  [3300, 900, 1050, 800, 1150, 1200, 960]);
const T4 = table('Effect of removing one sensor group at a time from LightGBM with extended features (leave-one-day-out, all seven days).',
  ['Sensor group removed', 'Recall', 'F1', 'False alarms', 'False alarms on 25 Dec'],
  ['none', 'sound', 'motion (PIR)', 'CO2', 'temperature'].map(k => [k === 'CO2' ? 'CO₂' : k, f3(abl[k].recall), f3(abl[k].F1), abl[k].FP, abl[k].FP_Dec25]),
  [2800, 1400, 1400, 1700, 2060]);
const T5 = table('Head-count estimation (0–3 people), evaluated on the three occupied days. Macro-F1 averages the F1 of the four classes equally.',
  ['Features', 'Model', 'Evaluation', 'Accuracy', 'Macro-F1'],
  hc.map(r => [r.features, r.model, r.evaluation, f3(r.accuracy), f3(r.macro_F1)]),
  [1900, 1900, 2400, 1600, 1560]);

const F1 = figure('fig1_jan10_lights_off.png', '10 January 2018, the only occupied test day in the original split. Panels from top: true number of people; light at desk nodes S1 and S2 (lux); CO₂ (ppm); motion (1 = either PIR sensor fired). The shaded block (15:25–17:00) marks the 184 snapshots that the logistic regression with light sensors classified as empty with near-certainty: three people were present, motion fired throughout and CO₂ rose from about 370 to 760 ppm, but the lights stayed near 7 lux.');
const F2 = figure('fig2_dec25_phantom_motion.png', '25 December 2017, a day on which the room was empty. Panels from top: firings of the door-side motion sensor S6 (seven firings; the three at 17:57–17:58 overlap); CO₂ (ppm); the loudest of the four sound sensors. The motion firings are not accompanied by any change in CO₂ or sound, so they are most likely spurious triggers.');
const F3 = figure('fig3_lodo_comparison.png', 'Leave-one-day-out comparison of selected models. Left: F1 for the occupied class. Right: false-alarm rate on empty snapshots (%). Orange bars are the models of the original analysis; blue bars are the re-analysis. Values are also listed in Table 3.');
const F6 = figure('fig6_permutation_importance.png', 'Permutation importance: the drop in F1 on the original test days (10–11 January) when one feature is randomly shuffled, averaged over 10 repeats. Left: the random forest of the original analysis. Right: LightGBM with extended features. Only the five most important features are shown.');
const F4 = figure('fig4_stillness.png', 'Stress test for a motionless occupant. In every occupied episode both motion sensors were set to zero for 0–40 minutes, starting 30 minutes into the episode; features were rebuilt and models re-evaluated leave-one-day-out on the three occupied days. The y-axis shows recall (share of occupied snapshots detected).');
const F5 = figure('fig5_release_latency.png', 'Time from the last occupied snapshot to the moment the room would be released, for each of the five departures in the dataset, under a rule that releases a room after 20 consecutive predicted-empty snapshots (nominally 10 minutes). Predictions are out-of-fold (leave-one-day-out).');
const F7 = figure('fig7_headcount_confusion.png', 'Head-count confusion matrix for a random forest using all features including light, evaluated leave-one-day-out on the three occupied days. Rows are true counts and are normalised to sum to 1; columns are predicted counts. Most errors confuse two and three people; 27% of three-person snapshots (the lights-off session) are predicted as empty.', 330);

// ---------- abstract ----------
const ABSTRACT = `Meeting rooms are often booked and then left empty, and buildings rarely know. I tested whether low-cost environmental sensors (temperature, light, sound, CO₂ and passive-infrared motion) can detect room occupancy without cameras or tracking individuals, using the public UCI Room Occupancy Estimation dataset (10,129 thirty-second snapshots over seven days). In my first analysis, a logistic regression trained on December data and tested on January data missed 184 of 294 occupied snapshots because three people worked with the lights off; the light sensors separated the training data almost perfectly, and removing them raised recall to 1.00. I then re-evaluated the work with leave-one-day-out cross-validation, compared 17 models and rules, and stress-tested the best candidates. The single train–test split had hidden a failure: the door-side motion sensor fired seven times on an empty holiday, and my final model raised ${LRo.FP_Dec25} false alarms on that day. A random forest given the light sensors failed exactly as the logistic regression did. Gradient boosting on backward-looking features that require corroborating evidence (motion counts, sound peaks, CO₂ rise above a trailing baseline) reached an F1 of ${f3(GB.F1)} with a ${pct(GB.FP_rate)} false-alarm rate, and kept recall above 0.97 when motion was silenced for 40 minutes. Head-count accuracy fell from ${pct(hcRandom.accuracy)} under a random split to ${pct(hcBest.accuracy)} under day-wise evaluation. Sensor-based occupancy detection is feasible, but only a day-wise evaluation shows how far it can be trusted.`;
const KEYWORDS = 'occupancy detection; environmental sensors; smart buildings; machine learning; cross-validation; shortcut learning';

// ---------- main text ----------
const body = [];
const placed = new Set();
body.push = function (...items) {           // after a paragraph, insert any table/figure it mentions for the first time
  for (const it of items) {
    Array.prototype.push.call(this, it);
    for (const m of (it._text || '').matchAll(/(Table|Figure) (\d+)/g)) {
      const key = m[0]; if (placed.has(key)) continue; placed.add(key);
      const i = Number(m[2]) - 1;
      Array.prototype.push.call(this, ...(m[1] === 'Table' ? renderTable(tables[i], i) : renderFigure(figures[i], i)));
    }
  }
  return this.length;
};
counting = 'body';
body.push(H1('1. Introduction'));
body.push(P(`Office buildings are designed around the assumption that rooms are used as booked, and they rarely are. A meeting room can show as occupied for an hour while standing empty, because a meeting was cancelled, moved or finished early. For the people looking for a room this is a daily frustration; for the building it is wasted space, cleaning effort and energy spent heating, cooling and ventilating nobody. If a room's own sensors could say reliably whether anyone is inside, a booking system could release "ghost" bookings, cleaning could be scheduled by use, and ventilation could follow real occupancy.`));
body.push(P(`Cameras and badge tracking can answer the question, but they raise privacy concerns and cost more to install. Environmental sensors (temperature, light, sound, CO₂ and passive-infrared (PIR) motion) are cheap, already common in buildings, and do not identify individuals. Earlier studies have shown that such sensors can detect or count occupants with high accuracy [1, 9]. Most of that work reports a single accuracy figure, however, and gives less attention to two questions that matter before anyone acts on the output: which sensors the model is really relying on, and whether the evaluation reflects the conditions the model will meet after deployment.`));
body.push(P(`This study started as my own data-science project on the UCI Room Occupancy Estimation dataset [1, 2]. The first analysis produced a clear result: the strongest sensor, light, made the model worse, because the model learned that "lights off" means "empty". Here I report that analysis and then a deeper re-analysis of it. My objectives were (i) to explain why the light-dependent model failed; (ii) to look for patterns in the data that the first analysis missed; (iii) to test whether the models, including the random forest, were built and evaluated well, using an evaluation that holds out whole days; (iv) to stress-test the best candidates against a motionless occupant and the loss of a sensor; and (v) to translate model output into the decision it would drive, namely when to release a room.`));

body.push(H1('2. Methods'));
body.push(H2('2.1 Dataset'));
body.push(P(`The UCI Room Occupancy Estimation dataset [1, 2] was recorded in a 6 m × 4.6 m room. Sensor nodes S1–S4 each carry a temperature, a light and a sound sensor; S5 carries a CO₂ sensor and also reports a CO₂ slope; S6 and S7 are ceiling PIR motion sensors, S6 by the door and S7 by the window. Readings arrive about every 30 seconds, and the head count (0–3 people) was recorded manually as ground truth. The data contain 10,129 snapshots across seven dates (${T1}) with no missing values. I defined a snapshot as occupied if at least one person was present, which gives 18.8% occupied snapshots.`));
body.push(H2('2.2 Original analysis'));
body.push(P(`I combined the date and time columns into a single timestamp and split the data by time: training on 22–26 December (8,084 snapshots, 19.9% occupied) and testing on 10–11 January (2,045 snapshots, 14.4% occupied). I split by time rather than at random because neighbouring 30-second snapshots are nearly identical, so a random split would place near-duplicates in both sets. I engineered two motion features: a 5-minute memory for each PIR sensor (the maximum over the last 10 snapshots) and the number of snapshots since the last motion. Because the costly error is releasing a room that people are using, I chose recall on the occupied class as the primary metric, with precision as a guardrail. I fitted logistic regression with and without class weighting, with and without the light sensors, and compared it with a random forest [3]. Finally, I simulated a policy that releases a room after N consecutive snapshots predicted as empty.`));
body.push(H2('2.3 Re-analysis: evaluation design'));
body.push(P(`For the re-analysis I used leave-one-day-out cross-validation: each of the seven days was held out in turn, the model was trained on the remaining six, and the out-of-fold predictions were pooled. This tests every occupied episode once and every empty day for false alarms, instead of one occupied episode. I report recall, precision and F1 for the occupied class, the false-alarm rate on empty snapshots, the area under the precision–recall curve, and the Brier score of predicted probabilities. All models used scikit-learn [5] or LightGBM [4].`));
body.push(H2('2.4 Features'));
body.push(P(`I recomputed all rolling features within continuous recording segments, splitting wherever the gap between snapshots exceeded five minutes, so that no window spans the 24-hour or 15-day gaps in the recording. In addition to the original features I built an extended set:`));
[ 'the number of motion firings in the last 5 and 10 minutes, rather than only whether any occurred;',
  'the peak and the mean of the loudest sound sensor over the last 5 minutes;',
  'the change in CO₂ over 5 and 10 minutes, and CO₂ above its trailing 2-hour minimum, a baseline that adapts to each day;',
  'a capped logarithm of the time since the last motion; and',
  'the 10-minute change in mean temperature.' ].forEach(t => body.push(BUL(t)));
body.push(P(`Every feature uses only past readings, so the model could run in real time. I deliberately excluded the time of day: occupancy in this dataset only occurs between 10:00 and 20:00, and a model given the clock could learn "offices are empty at night" instead of reading the sensors.`));
body.push(H2('2.5 Models and rules compared'));
body.push(P(`I compared 17 approaches: the original logistic regression and random forest, each with and without light; logistic regression with standardised inputs on the original, raw and extended features; random forests with a minimum leaf size of 20; LightGBM gradient boosting [4]; a support-vector machine with an RBF kernel; k-nearest neighbours; linear discriminant analysis; a depth-3 decision tree; and two transparent rules. The first rule flags a snapshot as occupied if either PIR sensor fired in the last 5 minutes. The second additionally requires corroboration: CO₂ at least 15 ppm above its 2-hour minimum, or a sound peak above 0.3 in the last 5 minutes. I set these thresholds after inspecting the data, so the rule results are somewhat optimistic.`));
body.push(H2('2.6 Stress tests and release policy'));
body.push(P(`To simulate a person sitting still, I set both PIR sensors to zero for 10–40 minutes starting 30 minutes into each occupied episode, rebuilt all features, and repeated the leave-one-day-out evaluation on the three occupied days. To test dependence on individual sensors, I removed one sensor group at a time (sound, motion, CO₂, temperature) from the best model. I also replayed the release policy on the out-of-fold predictions and measured, for each of the five departures in the data, the time between the last occupied snapshot and the moment the room would be released.`));
body.push(H2('2.7 What the models rely on, and head-count estimation'));
body.push(P(`I measured feature reliance with permutation importance on the original test days [6], and compared it with the models' built-in importance scores. Finally, I trained random forest and LightGBM classifiers to estimate the head count (0–3), evaluated both by holding out each occupied day and by a random 75/25 split, to show how much the evaluation choice changes the result [7, 8]. The complete analysis is a single notebook that runs from start to finish in about three minutes; it is available with the data and code (see Code availability).`));

body.push(H1('3. Results'));
body.push(H2('3.1 The original analysis and why the light sensors failed'));
body.push(P(`${T2} reproduces the original results. The logistic regression with all sensors missed 184 of 294 occupied test snapshots (recall 0.374). Class weighting and the time-since-motion feature left the confusion matrix unchanged, and lowering the decision threshold could not help, because every predicted probability was either below 0.01 or above 0.99. All 184 misses fall in one block on 10 January, 15:25–17:00 (${F1}): three people were present, motion fired and CO₂ climbed, but the lights stayed near 7 lux. Without the four light sensors the model missed nothing and raised 13 false alarms, all between 17:57 and 18:04, directly after the last occupied snapshot at 17:56:56. On its own training days it made 118 false alarms and one miss, so it was not simply memorising.`));
body.push(P(`The re-analysis explains why. In the training days "lights on" and "occupied" coincide almost exactly: the single rule S1_Light ≥ 50 misclassifies only 3 of 8,084 training snapshots, and a linear classifier on standardised features separates the training data with no errors. When classes are separable, logistic regression drives its coefficients towards infinity, which accounts for the saturated probabilities and for why class weights and thresholds had no effect. A random forest given the light sensors missed exactly the same 184 snapshots (${T2}). The problem therefore lies in the training data, which never contained people working in the dark, and not in the choice of algorithm.`));
body.push(H2('3.2 Patterns the first analysis missed'));
body.push(P(`**Phantom motion on an empty day.** On 25 December nobody was in the room, yet the door-side sensor S6 fired seven times, with no response in CO₂ or sound (${F2}). The original test days contained no such event. Under leave-one-day-out evaluation, the final model of the original analysis raised ${LRo.FP_Dec25} false alarms on that day alone, because each spurious firing switched on the 5-minute motion memory and reset the time since motion.`));
body.push(P(`**Only five occupied episodes.** All occupied time falls into five continuous episodes: two each on 22 and 23 December and one on 10 January. The original test set therefore contained one occupied episode and one departure, and its 100% recall describes a single event. A snapshot-level confidence interval would overstate the certainty, because consecutive snapshots are strongly autocorrelated.`));
body.push(P(`**The CO₂ slope is backward-looking.** The dataset's CO₂ slope correlates at r = ${Number(R.co2_slope_trailing25_corr).toFixed(4)} with an ordinary least-squares slope over the previous 25 snapshots (about 12.5 minutes), against r = 0.82 for a centred window of the same length. It therefore uses no future information, is safe for real-time use, and can be reproduced for any CO₂ stream.`));
body.push(P(`**Light behaves like desk lamps and daylight.** Mean light at S1 is about 120 lux whenever anyone is present, whereas S2 and S3 rise to 135–177 lux only with two or three people, so light encodes which desks are in use. On the empty 25 December, daylight raised S3 to 89 lux. The data never contain lights left on in an empty room, so that failure mode remains untested.`));
body.push(P(`**Errors cluster at arrivals and departures.** For the best model, ${40} of ${GB.FP} false alarms fall within 10 minutes of a labelled arrival or departure, when people move near the door and CO₂ lags. Some are labelling effects: on 10 January the motion sensors fired at 17:57:27 and 17:57:57, after the last snapshot labelled as occupied.`));
body.push(H2('3.3 Model comparison across all held-out days'));
body.push(P(`${T3} and ${F3} summarise the leave-one-day-out comparison. Three results stand out. First, the simple rule "motion in the last 5 minutes" (F1 ${f3(PIR5.F1)}) outperformed every model trained on the original features, including my final logistic regression (F1 ${f3(LRo.F1)}, false-alarm rate ${pct(LRo.FP_rate)}). Second, requiring corroboration from CO₂ or sound removed every false alarm on 25 December and raised F1 to ${f3(RULE.F1)}. Third, the best learned model was LightGBM on the extended features: recall ${f3(GB.recall)}, precision ${f3(GB.precision)}, F1 ${f3(GB.F1)}, a false-alarm rate of ${pct(GB.FP_rate)}, only ${GB.FP_Dec25} false alarms on 25 December, and the best-calibrated probabilities (Brier score ${GB.Brier}). Every model that was given the light sensors had a recall of 0.374 on 10 January.`));
body.push(H2('3.4 Assessment of the random forest'));
body.push(P(`The random forest in the original analysis was a reasonable first comparison, but the re-analysis shows five weaknesses. (i) The notebook used 1,000 trees while my project notes recorded 100; both give 11 false alarms on the original test days, and beyond roughly 100 trees additional trees only add computation. (ii) It was run only without light, although running it with light is the key control showing that the light failure is not specific to logistic regression. (iii) The trees were fully grown and untuned; across held-out days the unpruned forest had a recall of ${f3(RFo.recall)} (0.935 on 22 December), whereas requiring at least 20 samples per leaf raised recall to ${f3(L['RF regularised (min leaf 20)'].recall)}. (iv) It was judged on one episode: the original conclusion that it was "essentially identical" to logistic regression (11 against 13 false alarms) reverses across all days, where the forest raised ${RFo.FP} false alarms against ${LRo.FP}. (v) Its feature reliance was never examined. Its built-in importance ranks time since motion and motion memory highest, but permutation importance shows that its F1 depends mostly on the absolute CO₂ level (${F6}), which is fragile because baseline CO₂ changes with ventilation and outdoor air [6].`));
body.push(H2('3.5 Robustness to a motionless occupant and to sensor loss'));
body.push(P(`${F4} shows the stillness test. The corroborated rule depends on motion, so its recall fell steadily to ${f3(still(RULEname, 40).recall)} at 40 minutes of silence, with ${still(RULEname, 40).false_release_snapshots_N20} snapshots in which an occupied room would have been released. My original logistic regression was the most robust (recall ${f3(still(LRname, 40).recall)}, no false releases), because it also uses temperature and sound. LightGBM stayed at or above 0.968 recall throughout. Removing any single sensor group from LightGBM cost at most 0.023 in F1 (${T4}); sound mattered most, and without it the false alarms on 25 December rose from 6 to 83.`));
body.push(H2('3.6 Release policy'));
body.push(P(`Measured from each real departure (${F5}), a rule that releases a room after 20 consecutive predicted-empty snapshots, nominally 10 minutes, released rooms a median of ${pol(GBname, 20).median_release_after_departure_min} to ${pol(LRname, 20).median_release_after_departure_min} minutes after the last person left, because motion memory and CO₂ take several minutes to fade. With N = 10 the median delay was about 11 minutes, with no false releases for either the corroborated rule or the original logistic regression. The only false releases from LightGBM (3 snapshots) occurred directly after the 25-minute recording gap on 22 December, where the counter of empty predictions carried over from before the gap. On the original test day the policy could not have produced a false release, because the model had no misses there.`));
body.push(H2('3.7 What the models rely on'));
body.push(P(`Both the original random forest and LightGBM lean heavily on a single feature under permutation importance (${F6}): absolute CO₂ for the forest and the 5-minute sound peak for LightGBM. Because LightGBM loses only 0.023 in F1 when all sound features are removed (${T4}), this reflects which of several redundant signals it used rather than a single point of failure. If the time of day had been allowed as a feature, it would have ranked third in importance.`));
body.push(H2('3.8 Head-count estimation'));
body.push(P(`Estimating the number of people is much harder than detecting presence, and the answer depends strongly on the evaluation (${T5}). Under a random 75/25 split a random forest appeared almost perfect (accuracy ${f3(hcRandom.accuracy)}), and the same held for occupied-versus-empty detection (F1 0.993 against 0.943 under leave-one-day-out). Evaluated by held-out day, head-count accuracy ranged from 0.771 to ${f3(hcBest.accuracy)}, and most errors confused two and three people (${F7}). Unlike presence detection, counting benefits from light, because the desk lamps show how many desks are in use; the lights-off session still defeated it.`));

body.push(H1('4. Discussion'));
body.push(P(`The central lesson is that the most predictive feature in training can be the least trustworthy in deployment. Candanedo and Feldheim [9] found light to be the most important predictor of office occupancy, and my first analysis initially agreed. In this dataset, however, light was predictive because lights were switched on whenever people were present during the recorded days, not because light measures presence. A model that learns such a coincidence is an example of shortcut learning [10]: it performs well wherever the coincidence holds and fails, confidently, where it does not. The lights-off session exposed that shortcut, and removing light, or at least never letting light alone decide, is the appropriate response for presence detection.`));
body.push(P(`The second lesson concerns evaluation. The single chronological split was a sound choice, but it contained one occupied episode and no phantom motion, so it made the final model look safer than it is. Holding out each day in turn revealed ${LRo.FP_Dec25} false alarms that the original test could not see. At the other extreme, a random split, which lets near-duplicate snapshots appear in both training and test data, inflated head-count accuracy from ${pct(hcBest.accuracy)} to ${pct(hcRandom.accuracy)}. This is the kind of leakage described by Kaufman et al. [8], and grouping by day follows the recommendations of Roberts et al. [7] for data with temporal structure. Singh et al. [1], who collected this dataset, reported a maximum head-count accuracy of 98.4% and an F1 of 0.953. I could not verify their evaluation protocol, so my lower day-wise figures should be read as a more conservative estimate under a stricter evaluation, not as a contradiction. Their regression-based CO₂ slope was confirmed here to be a backward-looking 25-sample fit.`));
body.push(P(`The re-analysis also changes the practical recommendation. A transparent rule that requires motion together with a CO₂ rise or sound is attractive, but it fails when a person sits still. My original logistic regression tolerates stillness but is vulnerable to phantom motion. Gradient boosting on corroborated features balanced both in this data, with sensor redundancy that survived the loss of any one sensor group. Whatever the model, a release rule should state its real delay, which is about 11 minutes after departure at N = 10 and about 16 minutes at N = 20, reset after data gaps, and treat motion without supporting evidence as a possible sensor fault.`));
body.push(H2('Limitations'));
body.push(P(`The study covers one room, seven days and five occupied episodes, so all results are indicative rather than definitive. The extended features and rule thresholds were designed after exploring these data, and hyperparameters were not tuned in a nested loop, so the best-model figures are somewhat optimistic. The stillness test is a simulation: silencing the motion sensors leaves CO₂, sound and temperature as recorded, whereas a truly motionless person would probably also make less sound. The policy results are a backtest; no rooms were released. Finally, the ground truth was recorded manually, so the timing of arrivals and departures is uncertain by up to a minute or two.`));

body.push(H1('5. Conclusion'));
body.push(P(`Environmental sensors can detect room occupancy well without identifying anyone. On every held-out day, a gradient-boosted model on corroborated, backward-looking features detected ${pct(GB.recall)} of occupied snapshots with a ${pct(GB.FP_rate)} false-alarm rate and tolerated a simulated motionless occupant. The more lasting findings come from the failures: a strong sensor can encode a coincidence rather than a cause, a motion sensor can fire with nobody present, and an evaluation that tests one episode, or shuffles near-identical snapshots, can make a fragile model look perfect. I recommend piloting any such system with human confirmation before rooms are released automatically.`));
body.push(P(`Future work should collect many more occupied episodes across several rooms and seasons, deliberately including sessions with the lights off, lights left on in empty rooms, and long periods of stillness; evaluate models by room as well as by day; tune models in a nested, grouped cross-validation; and run a controlled pilot in which the release rule is actually applied, so that its effect on users can be measured rather than simulated.`));
counting = null;

// ---------- declarations and references ----------
const decl = [
  H1('Declarations'),
  PN('**Funding.** This work received no external funding.'),
  PN('**Conflicts of interest.** The author declares no conflicts of interest.'),
  PN('**Ethics approval.** This study used only a publicly available, de-identified dataset [2]. No new data were collected from people, and no ethics approval was required.'),
  PN('**Data availability.** The dataset is available from the UCI Machine Learning Repository (doi:10.24432/C5P605) under a CC BY 4.0 licence.'),
  PN(`**Code availability.** The notebooks and scripts that reproduce every result, table and figure are available at ${CODE_URL}.`),
  PN('**Use of AI tools.** I used Claude (Anthropic), an AI assistant, to learn and help write code wherever I needed it, and to help with grammar, wording and drafting of the text. I reviewed all code, results and text, and take full responsibility for the content.'),
  PN('**Acknowledgements.** I thank the creators of the Room Occupancy Estimation dataset for making it publicly available.'),
];
const REFS = [
  'A. P. Singh, V. Jain, S. Chaudhari, F. A. Kraemer, S. Werner and V. Garg, "Machine learning-based occupancy estimation using multivariate sensor nodes," in Proc. 2018 IEEE Globecom Workshops (GC Wkshps), Abu Dhabi, UAE, 2018. doi:10.1109/GLOCOMW.2018.8644432.',
  'A. P. Singh and S. Chaudhari, "Room Occupancy Estimation," UCI Machine Learning Repository, 2018. doi:10.24432/C5P605.',
  'L. Breiman, "Random forests," Machine Learning, vol. 45, no. 1, pp. 5–32, 2001.',
  'G. Ke et al., "LightGBM: A highly efficient gradient boosting decision tree," in Advances in Neural Information Processing Systems 30, 2017.',
  'F. Pedregosa et al., "Scikit-learn: Machine learning in Python," Journal of Machine Learning Research, vol. 12, pp. 2825–2830, 2011.',
  'C. Strobl, A.-L. Boulesteix, A. Zeileis and T. Hothorn, "Bias in random forest variable importance measures: Illustrations, sources and a solution," BMC Bioinformatics, vol. 8, art. 25, 2007.',
  'D. R. Roberts et al., "Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure," Ecography, vol. 40, no. 8, pp. 913–929, 2017.',
  'S. Kaufman, S. Rosset, C. Perlich and O. Stitelman, "Leakage in data mining: Formulation, detection, and avoidance," ACM Transactions on Knowledge Discovery from Data, vol. 6, no. 4, art. 15, 2012.',
  'L. M. Candanedo and V. Feldheim, "Accurate occupancy detection of an office room from light, temperature, humidity and CO2 measurements using statistical learning models," Energy and Buildings, vol. 112, pp. 28–39, 2016.',
  'R. Geirhos et al., "Shortcut learning in deep neural networks," Nature Machine Intelligence, vol. 2, pp. 665–673, 2020.',
];
const refs = [H1('References'), ...REFS.map((t, i) => new Paragraph({ spacing: DOUBLE, indent: { left: 567, hanging: 567 }, children: [new TextRun(`[${i + 1}]\t${t}`)] }))];

// ---------- title page and abstract ----------
const TITLE = 'When the Strongest Sensor Misleads: Privacy-Preserving Room Occupancy Detection from Environmental Sensors and the Evaluation Needed to Trust It';
counting = 'abstract'; count(ABSTRACT); counting = null;
const center = (children, after = 0) => new Paragraph({ alignment: AlignmentType.CENTER, spacing: { line: 300, after }, children });
const contact = `Email: ${EMAIL}` + (WITH_PHONE ? `  ·  Phone: ${PHONE}` : '');
const titlePage = [
  center([new TextRun({ text: TITLE, bold: true, size: 32 })], 360),
  center([new TextRun({ text: 'Prahlad Narayan Bhardwaj', bold: true, size: 26 })]),
  center([new TextRun(AFFILIATION)]),
  center([new TextRun(`ORCID: ${ORCID}`)]),
  center([new TextRun(contact)], 240),
  ...(WITH_PHONE ? [new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 240 }, children: [new TextRun({ text: `Abstract ${words.abstract} words  ·  main text ${words.body} words  ·  ${figures.length} figures  ·  ${tables.length} tables`, size: 20, color: '555555' })] })] : []),
  new Paragraph({ border: { top: { style: BorderStyle.SINGLE, size: 6, color: '999999', space: 8 } }, spacing: { before: 120, after: 60 }, children: [new TextRun({ text: 'Abstract', bold: true })] }),
  new Paragraph({ spacing: { line: 300, after: 120 }, children: runs(ABSTRACT) }),
  new Paragraph({ spacing: { line: 300, after: 120 }, border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: '999999', space: 8 } }, children: runs(`**Keywords:** ${KEYWORDS}.`) }),
];

const tail = [...refs];
tables.forEach((t, i) => { if (!placed.has(`Table ${i + 1}`)) tail.push(...renderTable(t, i)); });
figures.forEach((f, i) => { if (!placed.has(`Figure ${i + 1}`)) tail.push(...renderFigure(f, i)); });

const doc = new Document({
  creator: 'Prahlad Narayan Bhardwaj', title: TITLE,
  styles: {
    default: { document: { run: { font: FONT, size: 24 }, paragraph: { spacing: DOUBLE } } },
    paragraphStyles: [
      { id: 'Heading1', name: 'Heading 1', basedOn: 'Normal', next: 'Normal', quickFormat: true,
        run: { size: 24, bold: true, font: FONT }, paragraph: { spacing: { line: 360, before: 360, after: 120 }, outlineLevel: 0, keepNext: true } },
      { id: 'Heading2', name: 'Heading 2', basedOn: 'Normal', next: 'Normal', quickFormat: true,
        run: { size: 24, bold: true, italics: true, font: FONT }, paragraph: { spacing: { line: 360, before: 240, after: 60 }, outlineLevel: 1, keepNext: true } },
    ],
  },
  numbering: { config: [
    { reference: 'bullets', levels: [{ level: 0, format: LevelFormat.BULLET, text: '•', alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
  ] },
  sections: [{
    properties: {
      page: { size: { width: 12240, height: 15840 }, margin: { top: 1440, bottom: 1440, left: 1440, right: 1440, header: 720 } },
    },
    headers: { default: new Header({ children: [new Paragraph({ alignment: AlignmentType.RIGHT, children: [new TextRun({ children: [PageNumber.CURRENT] })] })] }) },
    children: [...titlePage, ...body, ...decl, ...tail],
  }],
});
Packer.toBuffer(doc).then(b => {
  fs.writeFileSync(path.join(__dirname, 'Room_Occupancy_Paper.docx'), b);
  console.log(`written: abstract ${words.abstract} words, body ${words.body} words, ${figures.length} figures, ${tables.length} tables`);
});
