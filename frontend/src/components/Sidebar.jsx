import { useCallback, useEffect, useState } from "react";
import { NavLink, useLocation } from "react-router-dom";
import {
  BarChart3,
  BrainCircuit,
  Database,
  LayoutDashboard,
  Menu,
  Shield,
  UserPlus,
  Video,
  X
} from "lucide-react";
import "./Sidebar.css";

const links = [
  { to: "/", label: "Dashboard", icon: LayoutDashboard },
  { to: "/live", label: "Live Monitor", icon: Video },
  { to: "/memory", label: "Memory", icon: Database },
  { to: "/reports", label: "Reports", icon: BarChart3 },
  { to: "/chat", label: "Chatbot", icon: BrainCircuit },
  { to: "/register", label: "Enroll Face", icon: UserPlus }
];

const MOBILE_MEDIA = "(max-width: 1100px)";

function Sidebar() {
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [isMobile, setIsMobile] = useState(() => window.matchMedia(MOBILE_MEDIA).matches);
  const location = useLocation();

  // Any navigation closes the drawer (link taps, back/forward, redirects).
  useEffect(() => {
    setDrawerOpen(false);
  }, [location]);

  // Escape closes; body scroll stays locked while the drawer is open on mobile.
  useEffect(() => {
    if (!drawerOpen) return undefined;
    const mobileQuery = window.matchMedia(MOBILE_MEDIA);
    if (!mobileQuery.matches) return undefined;

    const onKey = (event) => {
      if (event.key === "Escape") setDrawerOpen(false);
    };
    document.addEventListener("keydown", onKey);
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.removeEventListener("keydown", onKey);
      document.body.style.overflow = previousOverflow;
    };
  }, [drawerOpen]);

  // Leaving the mobile layout (rotate, resize) must not leave a frozen drawer.
  useEffect(() => {
    const mobileQuery = window.matchMedia(MOBILE_MEDIA);
    const onChange = (event) => {
      setIsMobile(event.matches);
      if (!event.matches) setDrawerOpen(false);
    };
    mobileQuery.addEventListener("change", onChange);
    return () => mobileQuery.removeEventListener("change", onChange);
  }, []);

  const closeDrawer = useCallback(() => setDrawerOpen(false), []);

  return (
    <>
      <header className="mobile-topbar">
        <button
          type="button"
          className="nav-burger"
          aria-label="Open navigation menu"
          aria-expanded={drawerOpen}
          aria-controls="sidebar-drawer"
          onClick={() => setDrawerOpen(true)}
        >
          <Menu size={22} />
        </button>
        <div className="mobile-topbar-brand">
          <Shield size={20} color="var(--accent-blue)" />
          <span>Vigilance AI</span>
        </div>
      </header>

      {drawerOpen && <div className="nav-scrim" onClick={closeDrawer} aria-hidden="true" />}

      <aside
        id="sidebar-drawer"
        className={`sidebar${drawerOpen ? " drawer-open" : ""}`}
        aria-hidden={!drawerOpen && isMobile}
      >
        <div className="sidebar-header">
          <Shield size={30} color="var(--accent-blue)" />
          <h2>Vigilance AI</h2>
          <button
            type="button"
            className="nav-drawer-close"
            aria-label="Close navigation menu"
            onClick={closeDrawer}
          >
            <X size={20} />
          </button>
        </div>
        <nav className="sidebar-nav" aria-label="Main navigation">
          {links.map((row) => {
            const Icon = row.icon;
            return (
              <NavLink
                key={row.to}
                to={row.to}
                end={row.to === "/"}
                className={({ isActive }) => `nav-item ${isActive ? "active" : ""}`}
              >
                <Icon size={18} />
                <span>{row.label}</span>
              </NavLink>
            );
          })}
        </nav>
        <div className="sidebar-footer">
          <div className="status-indicator">
            <div className="status-dot"></div>
            <span>Realtime API</span>
          </div>
        </div>
      </aside>
    </>
  );
}

export default Sidebar;
