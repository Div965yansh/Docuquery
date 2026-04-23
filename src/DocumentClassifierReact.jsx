

import React, { useState } from "react";

export default function DocumentClassifierReact() {
  const [text, setText] = useState("");
  const [fileName, setFileName] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const LABELS = ["World", "Sports", "Business", "Sci/Tech"];

  const handleFile = (e) => {
    const f = e.target.files[0];
    if (!f) return;
    setFileName(f.name);
    const reader = new FileReader();
    reader.onload = (ev) => {
      setText(ev.target.result);
    };
    reader.onerror = () => {
      setError("Failed to read file");
    };
    reader.readAsText(f);
  };

  const handleSubmit = async () => {
    setError(null);
    setResult(null);
    if (!text || text.trim().length === 0) {
      setError("Please provide some text or upload a file.");
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
        const body = await resp.text();
        throw new Error(`Server error: ${resp.status} - ${body}`);
      }

      const body = await resp.json();
      setResult(body);
    } catch (err) {
      setError(err.message || String(err));
    } finally {
      setLoading(false);
    }
  };

  const clearAll = () => {
    setText("");
    setFileName("");
    setResult(null);
    setError(null);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-100 via-purple-100 to-pink-100 flex items-center justify-center p-6">
      <div className="max-w-3xl w-full bg-white/80 backdrop-blur-xl rounded-3xl shadow-2xl p-8 border border-white/40">
        <h1 className="text-4xl font-bold mb-2 text-indigo-700 drop-shadow-sm">Document Classifier</h1>
        <p className="text-sm text-gray-600 mb-6">Upload a file or paste text to classify it using your trained BERT model.</p>

        <div className="mb-6">
          <label className="block text-sm font-medium text-gray-700 mb-2">Upload .txt file</label>
          <input
            type="file"
            accept=".txt"
            onChange={handleFile}
            className="block w-full text-sm text-gray-700 file:bg-indigo-600 file:text-white file:px-4 file:py-2 file:rounded-xl file:border-none file:cursor-pointer hover:file:bg-indigo-700 transition"
          />
          {fileName && <div className="text-xs text-gray-600 mt-1">Loaded: {fileName}</div>}
        </div>

        <div className="mb-6">
          <label className="block text-sm font-medium text-gray-700 mb-2">Or paste text</label>
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            rows={8}
            className="w-full p-4 border border-gray-300 rounded-xl shadow-sm focus:outline-none focus:ring-4 focus:ring-indigo-300 backdrop-blur-lg"
            placeholder="Paste text here..."
          />
        </div>

        <div className="flex gap-3 items-center mb-6">
          <button
            onClick={handleSubmit}
            disabled={loading}
            className="px-6 py-3 bg-indigo-600 text-white rounded-xl shadow-lg hover:bg-indigo-700 active:scale-95 transition disabled:opacity-60"
          >
            {loading ? "Classifying..." : "Classify"}
          </button>

          <button
            onClick={clearAll}
            className="px-6 py-3 bg-gray-200 text-gray-700 rounded-xl shadow hover:bg-gray-300 active:scale-95 transition"
          >
            Clear
          </button>

          <div className="ml-auto text-sm text-gray-600">Labels: {LABELS.join(', ')}</div>
        </div>

        {error && (
          <div className="mb-4 p-4 bg-red-100 text-red-700 border border-red-200 rounded-xl shadow">
            Error: {error}
          </div>
        )}

        {result && (
          <div className="mb-4 p-6 bg-green-50 border border-green-200 rounded-2xl shadow-lg">
            <div className="text-sm text-gray-600 mb-1 font-semibold">Prediction</div>
            <div className="text-2xl font-bold text-green-700">{result.predicted_category}</div>
            <div className="text-sm text-gray-700 mt-1">Confidence: {(result.confidence * 100).toFixed(2)}%</div>
            <div className="mt-4 text-xs text-gray-500 bg-white/50 p-3 rounded-xl max-h-48 overflow-auto">
              {result.text?.slice(0, 300)}{result.text && result.text.length > 300 ? '...' : ''}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
