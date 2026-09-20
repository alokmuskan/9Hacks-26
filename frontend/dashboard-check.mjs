// Verifies the dashboard card drill-downs and the readable event feed in a real
// browser, against a running backend (so the numbers and rows are real).
//
//   node dashboard-check.mjs            # against http://127.0.0.1:4173
//   BASE_URL=... node dashboard-check.mjs
//
// Exits non-zero if any check fails, so it can be wired into CI as-is.
import puppeteer from "puppeteer-core";

const CHROME =
  process.env.CHROME_PATH || `${process.env.LOCALAPPDATA}/Google/Chrome/Application/chrome.exe`;
const BASE = process.env.BASE_URL || "http://127.0.0.1:4173";

const browser = await puppeteer.launch({
  executablePath: CHROME,
  headless: true,
  args: ["--no-sandbox", "--disable-gpu"],
});

let failures = 0;
const check = (name, ok, detail = "") => {
  if (!ok) failures++;
  console.log(`${ok ? "ok  " : "FAIL"}  ${name}${detail ? `  (${detail})` : ""}`);
};

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

const page = await browser.newPage();
await page.setViewport({ width: 1440, height: 900 });
await page.goto(`${BASE}/`, { waitUntil: "networkidle2", timeout: 45000 });
await sleep(2500); // let the context finish its initial fetches

const text = (sel) => page.$eval(sel, (el) => el.textContent.trim()).catch(() => null);
const count = (sel) => page.$$eval(sel, (els) => els.length).catch(() => 0);
const exists = (sel) => page.$(sel).then((el) => Boolean(el));

// ── Cards ────────────────────────────────────────────────────────────────────
check("three metric cards render as buttons", (await count(".metric-card")) === 3);
const cardTags = await page.$$eval(".metric-card", (els) => els.map((el) => el.tagName));
check("cards are focusable buttons, not divs", cardTags.every((tag) => tag === "BUTTON"), cardTags.join(","));

const cardValues = await page.$$eval(".metric-copy div", (els) => els.map((el) => Number(el.textContent.trim())));
check(
  "card values are numeric and data-backed",
  cardValues.length === 3 && cardValues.every((n) => Number.isFinite(n)) && cardValues[0] >= 1,
  cardValues.join(" / ")
);

const hint = await text(".metric-hint");
check("cards advertise the drill-down", Boolean(hint && hint.includes("view details")), hint || "");

// ── Drill-down: Total Detections ─────────────────────────────────────────────
await page.click(".metric-card");
await sleep(500);
check("clicking a card opens the dialog", await exists(".detail-modal"));
const ariaModal = await page.$eval(".detail-modal", (el) => el.getAttribute("aria-modal")).catch(() => null);
check("dialog is aria-modal", ariaModal === "true");
check("body scroll locked while dialog open", await page.evaluate(() => document.body.style.overflow === "hidden"));

const tabs = await page.$$eval(".detail-tab", (els) => els.map((el) => el.textContent.trim()));
check("dialog exposes all three sections", tabs.length === 3, tabs.join(" | "));
check("detections tab is active by default", (await text(".detail-tab.active"))?.startsWith("Detections"));

const sessionRows = await count(".detail-table tbody tr");
check("detections tab lists recorded sessions", sessionRows >= 1, `${sessionRows} rows`);
const historyStats = await page.$$eval(".detail-stat", (els) =>
  els.map((el) => `${el.querySelector("span")?.textContent?.trim()}=${el.querySelector("strong")?.textContent?.trim()}`)
);
check(
  "detections tab shows live + recorded stats",
  historyStats.some((row) => row.startsWith("Sessions recorded")) && historyStats.some((row) => row.startsWith("Faces in frame"))
);

// ── Known people tab ─────────────────────────────────────────────────────────
await page.$$eval(".detail-tab", (els) => els.find((el) => el.textContent.includes("Known people"))?.click());
await sleep(400);
const knownActive = await text(".detail-tab.active");
check("known-people tab activates", Boolean(knownActive?.startsWith("Known people")), knownActive || "");
const knownRows = await count(".detail-table tbody tr");
const knownFirstCell = await text(".detail-table tbody tr td");
check("known people are listed with a name", knownRows >= 1 && Boolean(knownFirstCell), `${knownRows} rows, first="${knownFirstCell}"`);

// ── Unknown alerts tab ───────────────────────────────────────────────────────
await page.$$eval(".detail-tab", (els) => els.find((el) => el.textContent.includes("Unknown alerts"))?.click());
await sleep(400);
check("unknown-alerts tab activates", Boolean((await text(".detail-tab.active"))?.startsWith("Unknown alerts")));
const unknownStats = await page.$$eval(".detail-stat span", (els) => els.map((el) => el.textContent.trim()));
check(
  "unknown tab explains alerts and their rate",
  unknownStats.includes("Unknown alerts") && unknownStats.includes("Average per monitored minute"),
  unknownStats.slice(0, 4).join(" | ")
);
const unknownBody = await text(".detail-modal-body");
check(
  "unknown tab is never blank (table or explanation)",
  Boolean(unknownBody && (unknownBody.includes("Per minute") || unknownBody.includes("No unknown face"))),
  (unknownBody || "").slice(0, 60)
);

// ── Close behaviours ─────────────────────────────────────────────────────────
await page.keyboard.press("Escape");
await sleep(300);
check("Escape closes the dialog", !(await exists(".detail-modal")));

await page.click(".metric-card");
await sleep(400);
await page.click(".detail-modal-close");
await sleep(300);
check("close button closes the dialog", !(await exists(".detail-modal")));

await page.click(".metric-card");
await sleep(400);
await page.evaluate(() => {
  const scrim = document.querySelector(".detail-scrim");
  scrim.dispatchEvent(new MouseEvent("mousedown", { bubbles: true }));
});
await sleep(300);
check("scrim click closes the dialog", !(await exists(".detail-modal")));

// ── Readable event feed ──────────────────────────────────────────────────────
const headers = await page.$$eval(".log-table th", (els) => els.map((el) => el.textContent.trim()));
check("event feed uses plain-language headers", headers.join("|").includes("What happened"), headers.join(" / "));
const eventRows = await count(".log-table tbody tr");
const titles = await page.$$eval(".event-title", (els) => els.map((el) => el.textContent.trim()));
const summaries = await page.$$eval(".log-table tbody tr td:last-child", (els) => els.map((el) => el.textContent.trim()));
check("event feed has rows", eventRows >= 1, `${eventRows} rows`);
const friendly = titles.filter((t) =>
  /entered view|left view|Question asked|Monitoring session ended|Session summary|Face enrolled|Action|Memory searched|focus/.test(t)
);
check("event titles are sentences, not raw types", friendly.length >= 1, friendly.slice(0, 3).join(" | "));
const sentence = summaries.find((s) => /\.$/.test(s) && s.length > 25);
check("event details read as sentences", Boolean(sentence), (sentence || "").slice(0, 70));
check("rows keep the raw type for traceability", (await count(".event-type-badge")) >= 1);

// ── Responsive: no horizontal overflow, with dialog open ─────────────────────
for (const width of [390, 1440]) {
  await page.setViewport({ width, height: 900 });
  await sleep(400);
  const pageOverflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
  check(`no page overflow at ${width}px`, pageOverflow <= 0, `scrollW-innerW=${pageOverflow}`);

  await page.click(".metric-card");
  await sleep(400);
  const modalFits = await page.evaluate(() => {
    const modal = document.querySelector(".detail-modal");
    if (!modal) return null;
    const r = modal.getBoundingClientRect();
    return { left: Math.round(r.left), right: Math.round(r.right), overflow: document.documentElement.scrollWidth - window.innerWidth };
  });
  check(
    `dialog fits the viewport at ${width}px`,
    Boolean(modalFits && modalFits.left >= -1 && modalFits.right <= width + 1 && modalFits.overflow <= 0),
    JSON.stringify(modalFits)
  );
  await page.keyboard.press("Escape");
  await sleep(250);
}

await browser.close();
console.log(`\n${failures === 0 ? "ALL DASHBOARD CHECKS PASSED" : `${failures} CHECK(S) FAILED`}`);
process.exit(failures === 0 ? 0 : 1);
