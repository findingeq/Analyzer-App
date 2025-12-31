/**
 * VT Threshold Analyzer - Main App Component
 */

function App() {
  return (
    <div style={{ padding: "20px", backgroundColor: "#18181B", minHeight: "100vh" }}>
      <h1 style={{ color: "white", fontSize: "24px" }}>Test - Can you see this?</h1>
      <p style={{ color: "#a1a1aa" }}>If you can see this text, React is working.</p>
      <button
        style={{
          marginTop: "10px",
          padding: "10px 20px",
          backgroundColor: "#6366F1",
          color: "white",
          border: "none",
          borderRadius: "6px",
          cursor: "pointer"
        }}
        onClick={() => alert("Button clicked!")}
      >
        Click me
      </button>
    </div>
  );
}

export default App;
