// Verifies the generated detection review page actually works when clicked, because a
// broken button there costs the reviewer the exact minutes the page exists to save:
//   - every box renders with its index, class and thumbnail
//   - Correct / Wrong update the counters and the per-box state
//   - the downloaded payload carries the verdicts AND the optional missed-object notes
//   - an untouched box stays "unreviewed" (never silently counted as correct)
//   - the page states what it measures and what it does not
//
//   node review-check.mjs      # against ../reviews/review.html
//
// The page can be either blank or built with --verdicts (a correction pass). The blank
// assertions run against a copy whose only difference is an emptied verdict seed, so the
// same checks hold either way; the preload assertions run against the page as it is.
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import puppeteer from "puppeteer-core";

const CHROME =
  process.env.CHROME_PATH || `${process.env.LOCALAPPDATA}/Google/Chrome/Application/chrome.exe`;
const PAGE =
  process.env.REVIEW_PAGE || path.resolve(process.cwd(), "..", "reviews", "review.html");

if (!fs.existsSync(PAGE)) {
  console.error(`review page not found: ${PAGE}`);
  console.error("run:  python main.py review-detections");
  process.exit(1);
}

const fileUrl = (p) => `file://${p.replace(/\\/g, "/")}`;

let failures = 0;
const check = (label, ok, detail = "") => {
  if (ok) {
    console.log(`  PASS  ${label}`);
  } else {
    failures += 1;
    console.error(`  FAIL  ${label}${detail ? ` — ${detail}` : ""}`);
  }
};

// The page seeds its answers in one place, which is what makes a blank copy faithful.
const html = fs.readFileSync(PAGE, "utf8");
const seedMatch = html.match(/const verdicts = (\{[^;]*\});/);
if (!seedMatch) {
  console.error(`could not find the verdict seed in ${PAGE} — regenerate it with:\n` +
    "  python main.py review-detections");
  process.exit(1);
}
const seeded = JSON.parse(seedMatch[1]);
const preloadedCount = Object.keys(seeded).length;
const wrongWhenBuilt = Object.values(seeded).filter((value) => value === false).length;

const blankPath = path.join(os.tmpdir(), `review-check-blank-${process.pid}.html`);
fs.writeFileSync(blankPath, html.replace(seedMatch[0], "const verdicts = {};"), "utf8");

console.log(`page: ${PAGE}`);
console.log(
  preloadedCount > 0
    ? `built with --verdicts: ${preloadedCount} answer(s) already recorded\n`
    : "built blank: every box starts unanswered\n",
);

const browser = await puppeteer.launch({
  executablePath: CHROME,
  headless: true,
  args: ["--no-sandbox", "--disable-gpu"],
});

try {
  // --- Blank-page path, against the emptied seed ---
  const page = await browser.newPage();
  await page.goto(fileUrl(blankPath), { waitUntil: "load" });

  const rows = await page.$$(".row");
  check("every box is rendered as a row", rows.length > 0, `rows=${rows.length}`);

  const thumbs = await page.$$eval(".row img", (nodes) =>
    nodes.filter((n) => (n.getAttribute("src") || "").startsWith("data:image/jpeg;base64,")).length,
  );
  check("every row carries an embedded thumbnail", thumbs === rows.length, `${thumbs}/${rows.length}`);

  const indices = await page.$$eval(".row", (nodes) => nodes.map((n) => n.dataset.index));
  const unique = new Set(indices);
  check("indices are unique so a verdict is unambiguous", unique.size === indices.length);

  // Nothing touched yet: the page must not imply any box was judged.
  const before = await page.$eval(".progress", (n) => n.textContent);
  check("an untouched list counts zero reviewed", /reviewed 0 \//.test(before), before);

  // Mark the first two rows, one each way.
  const [first, second] = indices;
  await page.click(`.row[data-index="${first}"] button.yes`);
  await page.click(`.row[data-index="${second}"] button.no`);

  const after = await page.$eval(".progress", (n) => n.textContent);
  const wrong = await page.$eval(".wrongcount", (n) => n.textContent);
  check("clicking updates the reviewed counter", /reviewed 2 \//.test(after), after);
  check("clicking updates the wrong counter", /wrong 1/.test(wrong), wrong);

  const stateText = await page.$eval(`#state-${second}`, (n) => n.textContent);
  check("a rejected box shows as wrong", stateText.trim() === "wrong", stateText);

  const unmarked = await page.$eval(`#state-${indices[2]}`, (n) => n.textContent);
  check("an untouched box stays unreviewed", unmarked.trim() === "unreviewed", unmarked);

  // A click must move the cursor too. Otherwise the next keystroke lands on whichever
  // box the cursor was left on, not the one being judged — which is how a review ends
  // up with opposite answers on identical boxes.
  const cursor = await page.$eval(".row.active", (n) => n.dataset.index);
  check(
    "clicking moves the cursor to the next unreviewed box",
    cursor === indices[2],
    `${cursor} vs ${indices[2]}`,
  );

  // The optional miss note has to reach the payload, or the diagnostic list is empty.
  await page.type(`#missed-${first}`, "bottle, cup");

  const payload = JSON.parse(await page.evaluate(() => collect()));
  check("the payload records both verdicts", Object.keys(payload.verdicts).length === 2);
  check(
    "the payload carries the detection-list id so a stale review cannot be scored",
    typeof payload.fingerprint === "string" && payload.fingerprint.length === 16,
    JSON.stringify(payload.fingerprint),
  );
  check("the payload records the missed-object note", payload.missed[first] === "bottle, cup");
  check("the payload does not invent verdicts for untouched boxes", !(indices[2] in payload.verdicts));
  check(
    "verdicts are booleans, not strings the scorer has to guess at",
    payload.verdicts[first] === true && payload.verdicts[second] === false,
  );

  // --- Keyboard path, on a fresh blank page so the counters start clean ---
  await page.reload({ waitUntil: "load" });
  const keys = await page.$$eval(".row", (nodes) => nodes.map((n) => n.dataset.index));

  await page.keyboard.press("y");
  const afterY = await page.$eval(".progress", (n) => n.textContent);
  check("pressing Y marks the highlighted box", /reviewed 1 \//.test(afterY), afterY);
  const yState = await page.$eval(`#state-${keys[0]}`, (n) => n.textContent);
  check("Y records the box as correct", yState.trim() === "correct", yState);

  await page.keyboard.press("n");
  const afterN = await page.$eval(".progress", (n) => n.textContent);
  check("N marks the next box without a click", /reviewed 2 \//.test(afterN), afterN);
  const nState = await page.$eval(`#state-${keys[1]}`, (n) => n.textContent);
  check("N records the box as wrong", nState.trim() === "wrong", nState);

  const highlighted = await page.$eval(".row.active", (n) => n.dataset.index);
  check(
    "the highlight advances to the next unreviewed box",
    highlighted === keys[2],
    `${highlighted} vs ${keys[2]}`,
  );

  await page.click(`#missed-${keys[2]}`);
  await page.type(`#missed-${keys[2]}`, "yn");
  const afterTyping = await page.$eval(".progress", (n) => n.textContent);
  check("typing y/n in a note field is not read as a verdict", /reviewed 2 \//.test(afterTyping), afterTyping);

  // A sticky header would overlay the cards and swallow clicks on the top rows.
  const sticky = await page.$eval("header", (n) => getComputedStyle(n).position);
  check("the header does not stick over the cards", sticky !== "sticky", sticky);

  const footerCounters = await page.$$eval("footer .progress", (nodes) => nodes.length);
  check("the counters are mirrored at the end of the list", footerCounters === 1);

  const honesty = await page.$eval("header .honest", (n) => n.textContent.replace(/\s+/g, " "));
  check("the page states it measures precision only", /precision only/.test(honesty));
  check("the page states it cannot measure recall", /cannot measure recall/.test(honesty));
  check("the page states unreviewed is not correct", /never as correct/.test(honesty));

  const rubric = await page.$eval("details.rubric", (n) => n.textContent.replace(/\s+/g, " "));
  check("the page states the verdict criteria", /right object, right place/i.test(rubric));
  check("the rubric keeps a sloppy-on-the-right-object box correct", /still Correct/.test(rubric));
  check("the rubric says a duplicate box on one object is wrong", /One object, two boxes/.test(rubric));
  check("the rubric says label correctness is not about usefulness", /not.*about usefulness/.test(rubric));
  check("the rubric says an undecidable box stays unreviewed", /Leave it unreviewed/.test(rubric));

  const noindex = await page.$eval('meta[name="robots"]', (n) => n.content);
  check("the page is noindex because it shows a camera", /noindex/.test(noindex), noindex);

  // --- Correction pass, against the page as generated when it carries earlier answers ---
  if (preloadedCount > 0) {
    const pre = await browser.newPage();
    await pre.goto(fileUrl(PAGE), { waitUntil: "load" });

    const start = await pre.$eval(".progress", (n) => n.textContent);
    check(
      "a preloaded page opens at the previous answers",
      new RegExp(`reviewed ${preloadedCount} /`).test(start) &&
        new RegExp(`wrong ${wrongWhenBuilt}`).test(
          await pre.$eval(".wrongcount", (n) => n.textContent),
        ),
      start,
    );

    const painted = await pre.$$eval(".row.done", (nodes) => nodes.length);
    check(
      "preloaded verdicts are painted, not merely counted",
      painted === preloadedCount,
      `painted=${painted} of ${preloadedCount}`,
    );

    const stillActive = await pre.$eval(".row.active", (n) => n.dataset.index);
    const open = await pre.$$eval(".row:not(.done)", (nodes) =>
      nodes.length ? nodes[0].dataset.index : null,
    );
    check(
      "the cursor opens on the first box still unanswered",
      open === null || stillActive === open,
      `${stillActive} vs ${open}`,
    );

    // Correcting a box must replace that box's answer, not add a second one for it.
    const target = Object.keys(seeded)[0];
    const flipTo = seeded[target] === true ? "no" : "yes";
    await pre.click(`.row[data-index="${target}"] button.${flipTo}`);
    const corrected = await pre.$eval(".progress", (n) => n.textContent);
    check(
      "correcting a box replaces its answer rather than adding one",
      new RegExp(`reviewed ${preloadedCount} /`).test(corrected),
      corrected,
    );
    const flipped = await pre.$eval(`#state-${target}`, (n) => n.textContent);
    check(
      "the correction shows on the box it was made on",
      flipped.trim() === (flipTo === "yes" ? "correct" : "wrong"),
      flipped,
    );
  }
} finally {
  fs.rmSync(blankPath, { force: true });
  await browser.close();
}

console.log(failures === 0 ? "\nALL CHECKS PASSED" : `\n${failures} CHECK(S) FAILED`);
process.exit(failures === 0 ? 0 : 1);
