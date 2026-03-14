import "./LoadingScreen.css";

function LoadingScreen({ visible, title = "Loading", message = "Please wait..." }) {
  if (!visible) {
    return null;
  }

  return (
    <div className="loading-screen" role="status" aria-live="polite">
      <div className="loading-card">
        <div className="loading-spinner" aria-hidden="true"></div>
        <h3>{title}</h3>
        <p>{message}</p>
      </div>
    </div>
  );
}

export default LoadingScreen;
