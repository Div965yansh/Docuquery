import React, { useState } from "react";
import "./App.css";

const LABELS = ["World", "Sports", "Business", "Sci/Tech"];

export default function App() {
  const [text, setText] = useState("");
  const [file, setFile] = useState(null);
  const [fileName, setFileName] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  // Handle file selection
  const handleFileInput = (e) => {
    const f = e.target.files?.[0];
    if (!f) return;

    setFile(f);
    setFileName(f.name);
    setText("");
    setError("");
  };

  // Classify pasted text
  const submitText = async () => {
    setError("");
    setResult(null);

    if (!text.trim()) {
      setError("Please paste text or upload a document/image.");
      return;
    }

    setLoading(true);
    try {
      const resp = await fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });

      if (!resp.ok) {
        throw new Error(await resp.text());
      }

      setResult(await resp.json());
    } catch (err) {
      setError(err.message || "Something went wrong");
    } finally {
      setLoading(false);
    }
  };

  // Upload file
  const uploadFileToServer = async () => {
    setError("");
    setResult(null);

    if (!file) {
      setError("Please choose a file first.");
      return;
    }

    setLoading(true);
    try {
      const form = new FormData();
      form.append("file", file);

      const resp = await fetch("http://127.0.0.1:5000/predict_file", {
        method: "POST",
        body: form,
      });

      if (!resp.ok) {
        throw new Error(await resp.text());
      }

      setResult(await resp.json());
    } catch (err) {
      setError(err.message || "Something went wrong");
    } finally {
      setLoading(false);
    }
  };

  const clearAll = () => {
    setText("");
    setFile(null);
    setFileName("");
    setResult(null);
    setError("");
  };

  // Safe values
  const primaryCategory =
    result?.primary_category ||
    result?.predicted_category ||
    "Unknown";

  const confidenceValue =
    typeof result?.confidence === "number"
      ? result.confidence
      : 0;

  return (
    <div className="page-bg">
      <div className="center-card">
        <header className="card-header">
          <h1>DocSense — Document Classifier</h1>
          <p className="subtitle">
            Upload <strong>.txt / .pdf / .docx / .png / .jpg</strong>
            <br />
            or paste text and click <strong>Classify</strong>
          </p>
        </header>

        {/* Upload */}
        <div className="upload-row">
          <label className="file-button">
            Choose file
            <input
              type="file"
              accept=".txt,.pdf,.docx,.png,.jpg,.jpeg"
              onChange={handleFileInput}
            />
          </label>

          <div className="file-info">
            {fileName ? `Loaded: ${fileName}` : "No file selected"}
          </div>

          <button
            className="btn btn-accent"
            onClick={uploadFileToServer}
            disabled={loading}
          >
            {loading ? "Processing..." : "Upload & Classify"}
          </button>
        </div>

        {/* Paste text */}
        <label className="label">Or paste text</label>
        <textarea
          className="main-textarea"
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Paste document content here..."
        />

        {/* Actions */}
        <div className="action-row">
          <button
            className="btn btn-primary"
            onClick={submitText}
            disabled={loading}
          >
            {loading ? "Classifying..." : "Classify"}
          </button>

          <button className="btn btn-muted" onClick={clearAll}>
            Clear
          </button>

          <div className="labels">
            Labels: <strong>{LABELS.join(", ")}</strong>
          </div>
        </div>

        {/* Error */}
        {error && <div className="error-box">{error}</div>}

        {/* Result */}
        <div className="result-area">
          <h3>Last action</h3>

          {result ? (
            <div className="result-card">
              <div className="result-top">
                <div className="result-icon">🤖</div>
                <div>
                  <div className="result-cat">{primaryCategory}</div>

                  <div className="result-score">
                    Overall Confidence: {(confidenceValue * 100).toFixed(2)}%
                  </div>

                  {result.document_type && (
                    <div className="muted">
                      Document type:{" "}
                      <strong>{result.document_type}</strong> • Chunks:{" "}
                      <strong>{result.chunks_processed}</strong>
                    </div>
                  )}
                </div>
              </div>

              {/* Category distribution */}
              {Array.isArray(result.category_distribution) && (
                <div className="distribution">
                  <h4>Category Distribution</h4>
                  <ul>
                    {result.category_distribution.map((item) => (
                      <li key={item.label}>
                        {item.label}: {(item.confidence * 100).toFixed(1)}%
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {/* Preview */}
              <div className="preview">
                <strong>Extracted Text Preview:</strong>
                <pre>{result.text_preview || "—"}</pre>
              </div>
            </div>
          ) : (
            <div className="muted">
              No prediction yet. Upload a document/image or paste text.
            </div>
          )}
        </div>

        <footer className="card-footer">
          <div>
            Built for evaluation • Backend: <code>python app.py</code>
          </div>
        </footer>
      </div>
    </div>
  );
}
