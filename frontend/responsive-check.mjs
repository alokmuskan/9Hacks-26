import puppeteer from "puppeteer-core";

const CHROME =
  process.env.CHROME_PATH ||
  `${process.env.LOCALAPPDATA}/Google/Chrome/Application/chrome.exe`;

const BASE = process.env.BASE_URL || "http://127.0.0.1:4173";
const ROUTES = ["/", "/live", "/memory", "/reports", "/chat", "/register"];
const VIEWPORTS = [
  { name: "phone", width: 390, height: 844 },
  { name: "narrow-phone", width: 360, height: 800 },
  { name: "tablet", width: 820, height: 1180 },
  { name: "desktop", width: 1440, height: 900 },
];

const browser = await puppeteer.launch({
  executablePath: CHROME,
  headless: true,
  args: ["--no-sandbox", "--disable-gpu"],
});

let failures = 0;

for (const vp of VIEWPORTS) {
  const page = await browser.newPage();
  await page.setViewport({
    width: vp.width,
    height: vp.height,
    deviceScaleFactor: 2,
    isMobile: vp.width < 700,
    hasTouch: vp.width < 700,
  });
  for (const route of ROUTES) {
    await page.goto(BASE + route, { waitUntil: "networkidle2", timeout: 30000 });
    await new Promise((r) => setTimeout(r, 350));

    const m = await page.evaluate(() => {
      const de = document.documentElement;
      const docScrollW = Math.max(de.scrollWidth, document.body.scrollWidth);
      const innerW = window.innerWidth;
      // Elements that stick out past the viewport edge, excluding
      // intentional off-canvas pieces: the mobile nav (either the old
      // horizontal scroller or the closed slide-in drawer's subtree).
      const offenders = [];
      document.querySelectorAll("body *").forEach((el) => {
        if (el.closest(".sidebar-nav")) return;
        if (el.closest(".sidebar") && !el.classList.contains("sidebar")) {
          const sb = document.querySelector(".sidebar").getBoundingClientRect();
          if (sb.right <= 0) return; // closed drawer parked off-canvas by design
        }
        const r = el.getBoundingClientRect();
        if (
          r.width > 0 &&
          (r.right > innerW + 1 || r.left < -1) &&
          getComputedStyle(el).position !== "fixed"
        ) {
          offenders.push(
            `${el.tagName.toLowerCase()}.${String(el.className).split(" ")[0]}(right=${Math.round(r.right)},left=${Math.round(r.left)})`
          );
        }
      });
      return {
        docScrollW,
        innerW,
        overflowX: docScrollW > innerW + 1,
        offenders: offenders.slice(0, 4),
      };
    });

    const bad = m.overflowX || m.offenders.length > 0;
    if (bad) failures++;
    console.log(
      `${bad ? "FAIL" : "ok  "}  ${vp.name.padEnd(13)} ${vp.width}x${vp.height}  ${route.padEnd(9)} scrollW=${m.docScrollW} innerW=${m.innerW}` +
        (bad ? `  offenders: ${m.offenders.join(", ")}` : "")
    );
  }
  await page.close();
}

await browser.close();
console.log(failures === 0 ? "\nALL VIEWPORT CHECKS PASSED" : `\n${failures} CHECK(S) FAILED`);
process.exit(failures === 0 ? 0 : 1);
