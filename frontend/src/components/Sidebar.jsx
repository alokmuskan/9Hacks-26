import { NavLink } from "react-router-dom";
import {
  BarChart3,
  BrainCircuit,
  Database,
  LayoutDashboard,
  Shield,
  UserPlus,
  Video
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

function Sidebar() {
  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <Shield size={30} color="var(--accent-blue)" />
        <h2>Vigilance AI</h2>
      </div>
      <nav className="sidebar-nav">
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
  );
}

export default Sidebar;
