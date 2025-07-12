import React, { useState } from "react";
import "./App.css";

function App() {
  const [query, setQuery] = useState("");
  const [dataset, setDataset] = useState("wikIR1k");
  const [rankingMethod, setRankingMethod] = useState("tfidf");
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);

  const handleSearch = async (customQuery = null) => {
    const searchQuery = customQuery || query;
    if (!searchQuery.trim()) return;

    setLoading(true);
    setError(null);

    try {
      const response = await fetch(
        `http://127.0.0.1:8000/?q=${encodeURIComponent(
          searchQuery
        )}&dataset=${dataset}&method=${rankingMethod}`
      );
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      setResults(data);
      if (customQuery) setQuery(customQuery);
    } catch (err) {
      setError("An error occurred while fetching the data.");
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const groupByCluster = (docs) => {
    return docs.reduce((groups, doc) => {
      const cluster = doc.cluster || "Unclustered";
      if (!groups[cluster]) groups[cluster] = [];
      groups[cluster].push(doc);
      return groups;
    }, {});
  };

  const clusterColors = [
    "rgba(0, 245, 255, 0.12)",
    "rgba(0, 200, 255, 0.12)",
    "rgba(0, 180, 255, 0.10)",
    "rgba(0, 160, 255, 0.10)",
    "rgba(0, 140, 255, 0.08)",
    "rgba(0, 120, 255, 0.08)",
    "rgba(0, 100, 255, 0.06)",
    "rgba(0, 80, 255, 0.06)",
  ];

  return (
    <div className="App">
      <h1>IRProject Search System</h1>

      <input
        type="text"
        placeholder="Enter your search query..."
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        onKeyDown={(e) => e.key === "Enter" && handleSearch()}
      />

      <div style={{ marginTop: 15 }}>
        <label>
          <b>Select Dataset:</b>
        </label>
        <select
          value={dataset}
          onChange={(e) => setDataset(e.target.value)}
          style={{ marginLeft: "10px" }}
        >
          <option value="wikIR1k">wikIR1k</option>
          <option value="lifestyle">lifestyle</option>
        </select>

        <label style={{ marginLeft: "20px" }}>
          <b>Ranking Method:</b>
        </label>
        <select
          value={rankingMethod}
          onChange={(e) => setRankingMethod(e.target.value)}
          style={{ marginLeft: "10px" }}
        >
          <option value="tfidf">TF-IDF</option>
          <option value="bm25">BM25</option>
          <option value="hybrid">Hybrid</option>
        </select>
      </div>

      <button onClick={() => handleSearch()} style={{ marginTop: "20px" }}>
        Search
      </button>

      {loading && <p>Searching...</p>}
      {error && <p style={{ color: "red" }}>{error}</p>}

      {results && (
        <div style={{ marginTop: 30 }}>
          <h2>
            Search Results (
            {results.ranking_method
              ? results.ranking_method.toUpperCase()
              : "Unknown"}
            )
          </h2>
          <p>
            <b>Corrected Query:</b> {results.query_corrected}
          </p>

          {results.query_expanded && (
            <p>
              <b>Expanded Query:</b> {results.query_expanded}
            </p>
          )}

          <div style={{ marginTop: 15 }}>
            <h3>Evaluation Estimate</h3>
            <p>
              <b>Avg Similarity@10:</b>{" "}
              {results.evaluation_estimate?.["avg_similarity_percent@10"] ??
                "N/A"}
              %
            </p>
            <p>
              <b>Keyword Coverage:</b>{" "}
              {results.evaluation_estimate?.keyword_coverage ?? "N/A"}%
            </p>
            <p>
              <b>Matched Keywords:</b>{" "}
              {results.evaluation_estimate?.matched_keywords_total ?? "N/A"}
            </p>
            <p>
              <b>Inverted Index Terms Count:</b>{" "}
              {results.inverted_index_terms_count ?? "N/A"}
            </p>
          </div>

          {results.suggested_queries?.length > 0 && (
            <div style={{ marginTop: 20 }}>
              <h3>Suggested Queries</h3>
              <ul>
                {results.suggested_queries.map((suggestion, idx) => (
                  <li key={idx}>
                    <button
                      style={{
                        background: "none",
                        border: "none",
                        color: "#00f5ff",
                        cursor: "pointer",
                        textDecoration: "underline",
                      }}
                      onClick={() => handleSearch(suggestion)}
                    >
                      {suggestion}
                    </button>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {results.evaluation && (
            <div style={{ marginTop: 20 }}>
              <h3>Retrieval Evaluation</h3>
              <p>
                <b>Precision@10:</b>{" "}
                {results.evaluation["Precision@10"] ?? "N/A"}
              </p>
              <p>
                <b>Recall:</b> {results.evaluation["Recall"] ?? "N/A"}
              </p>
              <p>
                <b>MAP:</b> {results.evaluation["MAP"] ?? "N/A"}
              </p>
              <p>
                <b>MRR:</b> {results.evaluation["MRR"] ?? "N/A"}
              </p>
              {results.evaluation.note && (
                <p>
                  <i>{results.evaluation.note}</i>
                </p>
              )}
            </div>
          )}

          <h3>Top Matching Documents Grouped by Cluster:</h3>
          {results.top_similar_docs.length === 0 && <p>No results found.</p>}

          {Object.entries(groupByCluster(results.top_similar_docs)).map(
            ([cluster, docs], idx) => {
              const color = clusterColors[idx % clusterColors.length];
              return (
                <div
                  key={cluster}
                  style={{
                    backgroundColor: color,
                    padding: "15px",
                    marginBottom: "20px",
                    borderRadius: "10px",
                    border: "1px solid rgba(0, 245, 255, 0.25)",
                    boxShadow: "0 0 15px rgba(0, 245, 255, 0.2)",
                    transition: "transform 0.3s ease",
                  }}
                  onMouseEnter={(e) =>
                    (e.currentTarget.style.transform = "scale(1.03)")
                  }
                  onMouseLeave={(e) =>
                    (e.currentTarget.style.transform = "scale(1)")
                  }
                >
                  <h4 style={{ color: "#00f5ff", marginBottom: "10px" }}>
                    Cluster: {cluster} (Documents: {docs.length})
                  </h4>
                  <ul>
                    {docs.map((doc, i) => (
                      <li key={i} style={{ marginBottom: "10px" }}>
                        <p>
                          <b>Filename:</b> {doc.filename}
                        </p>
                        <p>
                          <b>Similarity Score:</b>{" "}
                          {doc.similarity_score
                            ? doc.similarity_score.toFixed(3)
                            : "N/A"}
                        </p>
                        <p>
                          <b>Dates Found:</b>{" "}
                          {doc.dates_found.length > 0
                            ? doc.dates_found.join(", ")
                            : "None"}
                        </p>
                        <p>
                          <b>Named Entities:</b>{" "}
                          {doc.named_entities.length > 0
                            ? doc.named_entities
                                .map((ent) => `${ent[0]} (${ent[1]})`)
                                .join(", ")
                            : "None"}
                        </p>
                      </li>
                    ))}
                  </ul>
                </div>
              );
            }
          )}
        </div>
      )}
    </div>
  );
}

export default App;
