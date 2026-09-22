import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import App from "./App";
import "./index.css";

// Every fresh page load starts on the dashboard, never on the last visited
// page: the browser keeps the URL across reloads, so rewrite it once here,
// before the router mounts. Module code runs exactly once per load (StrictMode
// double-invokes renders/effects, not module evaluation), so this cannot fire
// on in-app navigation. history.replaceState avoids a router transition —
// there is no flash and no extra history entry to go "back" into.
history.replaceState(null, "/", "/");

createRoot(document.getElementById("root")).render(
  <StrictMode>
    <BrowserRouter>
      <App />
    </BrowserRouter>
  </StrictMode>
);
