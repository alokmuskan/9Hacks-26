import { useEffect, useId, useRef } from "react";
import { X } from "lucide-react";
import "./DetailModal.css";

/**
 * Accessible detail dialog used by the dashboard cards.
 *
 * Follows the app's existing overlay patterns (the mobile nav drawer): scrim
 * click, Escape, background scroll lock, and focus moved into the dialog and
 * restored on close. Rendered only while `open`, so closed dialogs cost nothing.
 */
function DetailModal({ open, title, subtitle, tabs = [], activeTab, onTabChange, onClose, children }) {
  const titleId = useId();
  const closeRef = useRef(null);
  const restoreFocusRef = useRef(null);

  useEffect(() => {
    if (!open) {
      return undefined;
    }
    restoreFocusRef.current = document.activeElement;
    const onKeyDown = (event) => {
      if (event.key === "Escape") {
        event.stopPropagation();
        onClose?.();
      }
    };
    document.addEventListener("keydown", onKeyDown);
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    closeRef.current?.focus();
    return () => {
      document.removeEventListener("keydown", onKeyDown);
      document.body.style.overflow = previousOverflow;
      const restore = restoreFocusRef.current;
      if (restore && typeof restore.focus === "function") {
        restore.focus();
      }
    };
  }, [open, onClose]);

  if (!open) {
    return null;
  }

  return (
    <div
      className="detail-scrim"
      role="presentation"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) {
          onClose?.();
        }
      }}
    >
      <div className="detail-modal glass-panel" role="dialog" aria-modal="true" aria-labelledby={titleId}>
        <header className="detail-modal-head">
          <div className="detail-modal-titles">
            <h3 id={titleId}>{title}</h3>
            {subtitle ? <p className="detail-modal-sub">{subtitle}</p> : null}
          </div>
          <button
            type="button"
            ref={closeRef}
            className="detail-modal-close"
            onClick={onClose}
            aria-label="Close details"
          >
            <X size={18} />
          </button>
        </header>

        {tabs.length ? (
          <div className="detail-tabs" role="tablist" aria-label="Detail sections">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                type="button"
                role="tab"
                id={`detail-tab-${tab.id}`}
                aria-selected={tab.id === activeTab}
                aria-controls="detail-tabpanel"
                className={`detail-tab${tab.id === activeTab ? " active" : ""}`}
                onClick={() => onTabChange?.(tab.id)}
              >
                {tab.label}
                {tab.count === undefined || tab.count === null ? null : (
                  <span className="detail-tab-count">{tab.count}</span>
                )}
              </button>
            ))}
          </div>
        ) : null}

        <div className="detail-modal-body" id="detail-tabpanel" role="tabpanel">
          {children}
        </div>
      </div>
    </div>
  );
}

export default DetailModal;
