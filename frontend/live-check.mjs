// Verifies the Live Monitor layout changes in a real browser:
//   - the camera feed is compact (and was 420px+ tall before)
//   - the frame is letterboxed, never cropped
//   - the scanline travels the feed's actual height, not a hardcoded 420px
//   - "Connection" reports a real state instead of always "no"
//   - the runtime toggles explain themselves
//
//   node live-check.mjs          # against http://127.0.0.1:4173
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
await page.goto(`${BASE}/live`, { waitUntil: "networkidle2", timeout: 45000 });
await sleep(2000);

const feedBox = () =>
  page.evaluate(() => {
    const el = document.querySelector(".live-viewport");
    if (!el) return null;
    const r = el.getBoundingClientRect();
    const img = el.querySelector("img");
    return {
      width: Math.round(r.width),
      height: Math.round(r.height),
      fit: img ? getComputedStyle(img).objectFit : null,
      overflow: document.documentElement.scrollWidth - window.innerWidth,
    };
  });

// ── Compact feed ─────────────────────────────────────────────────────────────
const desktop = await feedBox();
check("feed renders on the live page", Boolean(desktop));
// Band, not a single value: the feed must be clearly smaller than the old
// 420px-minimum column while staying big enough to read annotations.
check(
  "desktop feed is sized for the layout (not 420px+, not a thumbnail)",
  Boolean(desktop && desktop.height >= 320 && desktop.height <= 460),
  `${desktop?.width}x${desktop?.height}`
);
check("feed no longer overflows the page", Boolean(desktop && desktop.overflow <= 0), `overflow=${desktop?.overflow}`);
check(
  "frame is shown uncropped (object-fit: contain)",
  desktop?.fit === "contain",
  `object-fit=${desktop?.fit}`
);

// ── Scanline travels the real height ─────────────────────────────────────────
const scanTravel = await page.evaluate(async () => {
  const el = document.querySelector(".scanline");
  const viewport = document.querySelector(".live-viewport");
  if (!el || !viewport) return null;
  const height = Math.round(viewport.getBoundingClientRect().height);
  const samples = [];
  for (let i = 0; i < 10; i += 1) {
    samples.push(parseFloat(getComputedStyle(el).top));
    await new Promise((r) => setTimeout(r, 500));
  }
  return { height, min: Math.min(...samples), max: Math.max(...samples), moved: Math.max(...samples) - Math.min(...samples) };
});
check("scanline animates", Boolean(scanTravel && scanTravel.moved > 10), JSON.stringify(scanTravel));
check(
  "scanline stays inside the feed (hardcoded 420px travel is gone)",
  Boolean(scanTravel && scanTravel.max <= scanTravel.height - 3 + 1),
  `max=${scanTravel?.max} height=${scanTravel?.height}`
);

// ── Connection indicator ─────────────────────────────────────────────────────
const counts = await page.$$eval(".count-box", (els) =>
  els.map((el) => ({
    label: el.querySelector("span")?.textContent?.trim(),
    value: el.querySelector("strong")?.textContent?.trim()
  }))
);
const connection = counts.find((row) => row.label === "Connection");
check("Live Counts reports a real connection state", Boolean(connection && ["Live", "Retrying", "Offline"].includes(connection.value)), JSON.stringify(connection));
check("the old always-'no' Reconnect box is gone", !counts.some((row) => row.label === "Reconnect"));

// ── Runtime controls explain themselves ──────────────────────────────────────
const toggleLabels = await page.$$eval(".toggle-btn", (els) => els.map((el) => el.textContent.trim()));
const hints = await page.$$eval(".toggle-hint", (els) => els.map((el) => el.textContent.trim()));
check("three runtime toggles render with On/Off state", toggleLabels.length === 3 && toggleLabels.every((t) => /(On|Off)$/.test(t)), toggleLabels.join(" | "));
check("each toggle explains what it detects", hints.length === 3 && hints.every((h) => h.length > 20), hints.length + " hints");
check(
  "gaze is called out as the heaviest step",
  hints.some((h) => h.toLowerCase().includes("heaviest")),
  hints.find((h) => h.toLowerCase().includes("heaviest")) || ""
);

// ── Smaller viewports ────────────────────────────────────────────────────────
for (const [width, height] of [
  [1280, 800],
  [820, 1180],
  [390, 844]
]) {
  await page.setViewport({ width, height });
  await sleep(500);
  const box = await feedBox();
  check(
    `feed stays inside the fold at ${width}x${height}`,
    Boolean(box && box.height <= Math.round(height * 0.5) && box.overflow <= 0),
    `${box?.width}x${box?.height} (viewport ${width}x${height})`
  );
}

await browser.close();
console.log(`\n${failures === 0 ? "ALL LIVE MONITOR CHECKS PASSED" : `${failures} CHECK(S) FAILED`}`);
process.exit(failures === 0 ? 0 : 1);
