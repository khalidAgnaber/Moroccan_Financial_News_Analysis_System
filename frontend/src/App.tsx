import React from 'react';
import { useAuth0 } from '@auth0/auth0-react';
import LandingPage from './components/LandingPage';
import Dashboard from './components/Dashboard';
import './App.css';

function App() {
  const { isAuthenticated, isLoading } = useAuth0();

  if (isLoading) {
    return (
      <div className="loading-container">
        <div className="loading-spinner"></div>
        <p>Chargement de l'application...</p>
      </div>
    );
  }

  return (
    <div className="App">
      {isAuthenticated ? <Dashboard /> : <LandingPage />}
    </div>
  );
}

export default App;