// src/components/Dashboard.tsx
import React, { useState, useEffect } from 'react';
import { useAuth0 } from '@auth0/auth0-react';
import './Dashboard.css';

interface AnalysisResult {
  text: string;
  sentiment: 'positive' | 'negative' | 'neutral' | null;
  timestamp: Date;
}

interface NewsItem {
  id: string | number;
  title: string;
  text: string;
  source: string;
  url: string;
  date: string;
  sentiment?: 'positive' | 'negative' | 'neutral' | null;  // safer
  branch?: string | null;
}

const Dashboard: React.FC = () => {
  const { user, logout } = useAuth0();

  const [activeTab, setActiveTab] = useState<'news' | 'analyze'>('news');
  const [todaysNews, setTodaysNews] = useState<NewsItem[]>([]);
  const [expandedIds, setExpandedIds] = useState<Set<string | number>>(new Set());

  const [newsText, setNewsText] = useState('');
  const [analysis, setAnalysis] = useState<AnalysisResult | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);

  useEffect(() => {
    const fetchNews = async () => {
      setLoadError(null);
      try {
        const res = await fetch('/api/news');
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data: NewsItem[] = await res.json();

        console.log('Fetched /api/news (first 5 rows):', data.slice(0, 5));

        const sorted = data.sort(
          (a, b) =>
            new Date(b.date).getTime() - new Date(a.date).getTime()
        );
        setTodaysNews(sorted);
      } catch (err: any) {
        console.error('Erreur lors du chargement des actualités:', err);
        setTodaysNews([]);
        setLoadError(err.message ?? String(err));
      }
    };

    fetchNews();
    const interval = setInterval(fetchNews, 15 * 60 * 1000);
    return () => clearInterval(interval);
  }, []);

  const toggleExpand = (id: string | number) => {
    const next = new Set(expandedIds);
    next.has(id) ? next.delete(id) : next.add(id);
    setExpandedIds(next);
  };

  const handleAnalyze = async () => {
    if (!newsText.trim()) return;
    setIsAnalyzing(true);
    setAnalysis(null);
    try {
      const res = await fetch('/api/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: newsText })
      });
      const json = await res.json();
      setAnalysis({
        text: json.text,
        sentiment: json.sentiment ?? 'neutral',
        timestamp: new Date(json.timestamp)
      });
      setNewsText('');
    } catch (e) {
      console.error('Analysis failed:', e);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const getBadge = (sentiment: string | null | undefined) => {
    switch (sentiment) {
      case 'positive':
        return { icon: '📈', color: '#10B981' };
      case 'negative':
        return { icon: '📉', color: '#EF4444' };
      case 'neutral':
      default:
        return { icon: '➖', color: '#6B7280' };
    }
  };

  return (
    <div className="dashboard-container">
      <header className="dashboard-header">
        <div className="header-left">
          <h1>Analyse de Sentiment Financier</h1>
          <p>Actualités financières marocaines</p>
        </div>
        <div className="header-right">
          <span>👤 {user?.name}</span>
          <button
            className="logout-button"
            onClick={() =>
              logout({ logoutParams: { returnTo: window.location.origin } })
            }
          >
            Déconnexion
          </button>
        </div>
      </header>

      <nav className="dashboard-nav">
        <button
          className={`nav-tab${activeTab === 'news' ? ' active' : ''}`}
          onClick={() => setActiveTab('news')}
        >
          📺 Actualités du jour
        </button>
        <button
          className={`nav-tab${activeTab === 'analyze' ? ' active' : ''}`}
          onClick={() => setActiveTab('analyze')}
        >
          🔍 Analyser
        </button>
      </nav>

      <main className="dashboard-main">
        {activeTab === 'news' && (
          <section className="news-section">
            {loadError && (
              <p className="error-text">
                Erreur lors du chargement des actualités: {loadError}
              </p>
            )}
            {!loadError && todaysNews.length === 0 && <p>Chargement…</p>}

            <div className="news-grid">
              {todaysNews.map((item) => {
                const sentiment = item.sentiment ?? 'neutral';
                const { icon, color } = getBadge(sentiment);
                const safeId = item.id ?? `${item.title}-${item.date}`;
                const isExpanded = expandedIds.has(safeId);

                return (
                  <div key={safeId} className="news-card">
                    <div className="news-header">
                      <span className="news-date">
                        {new Date(item.date).toLocaleString('fr-FR')}
                      </span>

                      <div
                        style={{
                          display: 'flex',
                          gap: '0.5rem',
                          alignItems: 'center'
                        }}
                      >
                        <span
                          className="sentiment-badge"
                          style={{ backgroundColor: color }}
                        >
                          {icon} {sentiment.toUpperCase()}
                        </span>

                        <span className="branch-badge">
                          {item.branch ? String(item.branch) : '—'}
                        </span>
                      </div>
                    </div>

                    <h3 className="news-title">{item.title}</h3>

                    <p
                      className={`news-text${isExpanded ? '' : ' truncated'}`}
                    >
                      {item.text}
                    </p>

                    {!isExpanded ? (
                      <button
                        className="read-more"
                        onClick={() => toggleExpand(safeId)}
                      >
                        Lire la suite
                      </button>
                    ) : (
                      <button
                        className="read-more"
                        onClick={() => toggleExpand(safeId)}
                      >
                        Afficher moins
                      </button>
                    )}

                    <p className="news-source">
                      Source:{' '}
                      <a
                        href={item.url}
                        target="_blank"
                        rel="noopener noreferrer"
                      >
                        {item.source}
                      </a>
                    </p>
                  </div>
                );
              })}
            </div>
          </section>
        )}

        {activeTab === 'analyze' && (
          <section className="analyze-section">
            <div className="input-section">
              <textarea
                className="news-input"
                value={newsText}
                onChange={(e) => setNewsText(e.target.value)}
                placeholder="Collez une actualité financière ici…"
                rows={6}
              />
              <button
                className="analyze-button"
                onClick={handleAnalyze}
                disabled={isAnalyzing}
              >
                {isAnalyzing ? 'Analyse en cours…' : 'Analyser le sentiment'}
                {isAnalyzing && <span className="spinner" />}
              </button>
            </div>
            {analysis && (
              <div className="analysis-result">
                <h3>Résultat d’Analyse</h3>
                <p>{analysis.text}</p>
                <div
                  className="sentiment-badge"
                  style={{
                    backgroundColor: getBadge(analysis.sentiment).color
                  }}
                >
                  {getBadge(analysis.sentiment).icon}{' '}
                  {(analysis.sentiment ?? 'neutral').toUpperCase()}
                </div>
                <small>{analysis.timestamp.toLocaleString('fr-FR')}</small>
              </div>
            )}
          </section>
        )}
      </main>
    </div>
  );
};

export default Dashboard;
