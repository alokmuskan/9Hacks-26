import puppeteer from "puppeteer-core";

const CHROME =
  process.env.CHROME_PATH ||
  `${process.env.LOCALAPPDATA}/Google/Chrome/Application/chrome.exe`;
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

// ── Mobile behavior ──────────────────────────────────────────────────────────
{
  const page = await browser.newPage();
  await page.setViewport({ width: 390, height: 844, deviceScaleFactor: 2, isMobile: true, hasTouch: true });
  await page.goto(`${BASE}/`, { waitUntil: "networkidle2", timeout: 30000 });

  const visible = (sel) =>
    page.evaluate((s) => {
      const el = document.querySelector(s);
      if (!el) return false;
      const st = getComputedStyle(el);
      const r = el.getBoundingClientRect();
      return st.display !== "none" && st.visibility !== "hidden" && r.width > 0 && r.height > 0;
    }, sel);

  const drawerX = () =>
    page.evaluate(() => {
      const r = document.querySelector(".sidebar").getBoundingClientRect();
      return { left: Math.round(r.left), right: Math.round(r.right), width: Math.round(r.width) };
    });

  // Initial state: topbar visible, drawer off-canvas, no scrim.
  check("mobile: topbar visible", await visible(".mobile-topbar"));
  check("mobile: burger visible", await visible(".nav-burger"));
  check("mobile: scrim hidden initially", !(await visible(".nav-scrim")));
  let x = await drawerX();
  check("mobile: drawer off-canvas initially", x.right <= 0, `right=${x.right}`);

  await page.click(".nav-burger");
  await new Promise((r) => setTimeout(r, 400)); // slide transition
  x = await drawerX();
  check("mobile: drawer slides in on burger tap", x.left === 0 && x.width > 200, `left=${x.left} w=${x.width}`);
  check("mobile: scrim visible when open", await visible(".nav-scrim"));
  const ariaOpen = await page.$eval(".nav-burger", (el) => el.getAttribute("aria-expanded"));
  check("mobile: aria-expanded=true when open", ariaOpen === "true");

  // Body scroll locked while open.
  const locked = await page.evaluate(() => document.body.style.overflow === "hidden");
  check("mobile: body scroll locked while open", locked);

  // Escape closes.
  await page.keyboard.press("Escape");
  await new Promise((r) => setTimeout(r, 400));
  x = await drawerX();
  check("mobile: Escape closes drawer", x.right <= 0, `right=${x.right}`);
  const unlocked = await page.evaluate(() => document.body.style.overflow !== "hidden");
  check("mobile: scroll unlocked after close", unlocked);

  // Scrim tap closes.
  await page.click(".nav-burger");
  await new Promise((r) => setTimeout(r, 350));
  await page.click(".nav-scrim");
  await new Promise((r) => setTimeout(r, 400));
  x = await drawerX();
  check("mobile: scrim tap closes drawer", x.right <= 0, `right=${x.right}`);

  // Close (X) button closes.
  await page.click(".nav-burger");
  await new Promise((r) => setTimeout(r, 350));
  await page.click(".nav-drawer-close");
  await new Promise((r) => setTimeout(r, 400));
  x = await drawerX();
  check("mobile: X button closes drawer", x.right <= 0, `right=${x.right}`);

  // Tapping a link navigates and closes the drawer.
  await page.click(".nav-burger");
  await new Promise((r) => setTimeout(r, 350));
  await page.click('.sidebar-nav a[href="/live"]');
  await new Promise((r) => setTimeout(r, 500));
  x = await drawerX();
  const path = await page.evaluate(() => window.location.pathname);
  check("mobile: link tap navigates", path === "/live", `path=${path}`);
  check("mobile: link tap closes drawer", x.right <= 0, `right=${x.right}`);

  await page.close();
}

// ── Desktop behavior unchanged ───────────────────────────────────────────────
{
  const page = await browser.newPage();
  await page.setViewport({ width: 1440, height: 900 });
  await page.goto(`${BASE}/`, { waitUntil: "networkidle2", timeout: 30000 });

  const state = await page.evaluate(() => {
    // An element that is absent from the DOM is hidden by definition.
    const st = (s) => {
      const el = document.querySelector(s);
      return el ? getComputedStyle(el).display : "none";
    };
    const r = document.querySelector(".sidebar").getBoundingClientRect();
    return {
      topbar: st(".mobile-topbar"),
      burger: st(".nav-burger"),
      close: st(".nav-drawer-close"),
      scrim: st(".nav-scrim"),
      sidebarLeft: Math.round(r.left),
      ariaHidden: document.querySelector(".sidebar").getAttribute("aria-hidden"),
    };
  });
  check("desktop: topbar hidden", state.topbar === "none");
  check("desktop: burger hidden", state.burger === "none");
  check("desktop: close btn hidden", state.close === "none");
  check("desktop: scrim hidden", state.scrim === "none");
  check("desktop: sidebar pinned left", state.sidebarLeft === 0, `left=${state.sidebarLeft}`);
  check("desktop: sidebar not aria-hidden", state.ariaHidden !== "true");

  await page.close();
}

await browser.close();
console.log(failures === 0 ? "\nALL DRAWER CHECKS PASSED" : `\n${failures} DRAWER CHECK(S) FAILED`);
process.exit(failures === 0 ? 0 : 1);
