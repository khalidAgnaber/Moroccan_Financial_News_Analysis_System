import React from 'react';
import { useAuth0 } from '@auth0/auth0-react';
import './LandingPage.css';

const LandingPage: React.FC = () => {
  const { loginWithRedirect, isAuthenticated, isLoading } = useAuth0();

  const handleLogin = () => {
    loginWithRedirect();
  };

  if (isLoading) {
    return (
      <div className="loading-container">
        <div className="loading-spinner"></div>
        <p>Chargement...</p>
      </div>
    );
  }

  if (isAuthenticated) {
    return (
      <div className="authenticated-container">
        <h2>Bienvenue dans l'analyse de sentiment financier</h2>
        <p>Vous êtes connecté avec succès!</p>
      </div>
    );
  }

  return (
    <div className="landing-container">
      <div className="background-animation">
        <div className="floating-shapes">
          <div className="shape shape-1"></div>
          <div className="shape shape-2"></div>
          <div className="shape shape-3"></div>
          <div className="shape shape-4"></div>
          <div className="shape shape-5"></div>
          <div className="shape shape-6"></div>
        </div>
      </div>

      <div className="top-logo">
        <img 
          src="/logo-redmed.png"
          alt="Red Med Capital" 
          className="company-logo-small"
        />
        <span className="company-name-small">Red Med Capital</span>
      </div>

      <div className="main-content">
        <div className="hero-section">
          <h1 className="hero-title">
            Analyse de Sentiment des Actualités Financières Marocaines
          </h1>
          <p className="hero-description">
            Plateforme d'analyse intelligente pour évaluer le sentiment des actualités 
            financières marocaines en temps réel
          </p>
        </div>

        <div className="login-section">
          <div className="login-card">
            <h3>Connexion</h3>
            <p className="login-subtitle">
              Accédez à votre tableau de bord d'analyse de sentiment
            </p>
            
            <button 
              className="login-button"
              onClick={handleLogin}
            >
              <svg className="login-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 16l-4-4m0 0l4-4m-4 4h14m-5 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013 3v1" />
              </svg>
              Se connecter
            </button>

          </div>
        </div>
      </div>

      <footer className="landing-footer">
        <p>&copy; 2025 Red Med Capital - Analyse de Sentiment Financier</p>
      </footer>
    </div>
  );
};

export default LandingPage;