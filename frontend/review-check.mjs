// Verifies the generated detection review page actually works when clicked, because a
// broken button there costs the reviewer the exact minutes the page exists to save:
//   - every box renders with its index, class and thumbnail
//   - Correct / Wrong update the counters and the per-box state
//   - the downloaded payload carries the verdicts AND the optional missed-object notes
//   - an untouched box stays "unreviewed" (never silently counted as correct)
//   - the page states what it measures and what it does not
//
//   node review-check.mjs      # against ../reviews/review.html
import fs from "node:fs";
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

let failures = 0;
const check = (label, ok, detail = "") => {
  if (ok) {
    console.log(`  PASS  ${label}`);
  } else {
    failures += 1;
    console.error(`  FAIL  ${label}${detail ? ` — ${detail}` : ""}`);
  }
};

const browser = await puppeteer.launch({
  executablePath: CHROME,
  headless: true,
  args: ["--no-sandbox", "--disable-gpu"],
});

try {
  const page = await browser.newPage();
  await page.goto(`file://${PAGE.replace(/\\/g, "/")}`, { waitUntil: "load" });

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
  const before = await page.$eval("#progress", (n) => n.textContent);
  check("an untouched list counts zero reviewed", /reviewed 0 \//.test(before), before);

  // Mark the first two rows, one each way.
  const [first, second] = indices;
  await page.click(`.row[data-index="${first}"] button.yes`);
  await page.click(`.row[data-index="${second}"] button.no`);

  const after = await page.$eval("#progress", (n) => n.textContent);
  const wrong = await page.$eval("#wrongcount", (n) => n.textContent);
  check("clicking updates the reviewed counter", /reviewed 2 \//.test(after), after);
  check("clicking updates the wrong counter", /wrong 1/.test(wrong), wrong);

  const stateText = await page.$eval(`#state-${second}`, (n) => n.textContent);
  check("a rejected box shows as wrong", stateText.trim() === "wrong", stateText);

  const unmarked = await page.$eval(`#state-${indices[2]}`, (n) => n.textContent);
  check("an untouched box stays unreviewed", unmarked.trim() === "unreviewed", unmarked);

  // The optional miss note has to reach the payload, or the diagnostic list is empty.
  await page.type(`#missed-${first}`, "bottle, cup");

  const payload = JSON.parse(await page.evaluate(() => collect()));
  check("the payload records both verdicts", Object.keys(payload.verdicts).length === 2);
  check("the payload records the missed-object note", payload.missed[first] === "bottle, cup");
  check("the payload does not invent verdicts for untouched boxes", !(indices[2] in payload.verdicts));
  check(
    "verdicts are booleans, not strings the scorer has to guess at",
    payload.verdicts[first] === true && payload.verdicts[second] === false,
  );

  // --- Keyboard path, on a fresh page so the counters start clean ---
  await page.reload({ waitUntil: "load" });
  const keys = await page.$$eval(".row", (nodes) => nodes.map((n) => n.dataset.index));

  await page.keyboard.press("y");
  const afterY = await page.$eval("#progress", (n) => n.textContent);
  check("pressing Y marks the highlighted box", /reviewed 1 \//.test(afterY), afterY);
  const yState = await page.$eval(`#state-${keys[0]}`, (n) => n.textContent);
  check("Y records the box as correct", yState.trim() === "correct", yState);

  await page.keyboard.press("n");
  const afterN = await page.$eval("#progress", (n) => n.textContent);
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
  const afterTyping = await page.$eval("#progress", (n) => n.textContent);
  check("typing y/n in a note field is not read as a verdict", /reviewed 2 \//.test(afterTyping), afterTyping);

  const honesty = await page.$eval("header .honest", (n) => n.textContent.replace(/\s+/g, " "));
  check("the page states it measures precision only", /precision only/.test(honesty));
  check("the page states it cannot measure recall", /cannot measure recall/.test(honesty));
  check("the page states unreviewed is not correct", /never as correct/.test(honesty));

  const noindex = await page.$eval('meta[name="robots"]', (n) => n.content);
  check("the page is noindex because it shows a camera", /noindex/.test(noindex), noindex);
} finally {
  await browser.close();
}

console.log(failures === 0 ? "\nALL CHECKS PASSED" : `\n${failures} CHECK(S) FAILED`);
process.exit(failures === 0 ? 0 : 1);
