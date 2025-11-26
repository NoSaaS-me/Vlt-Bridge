import { useEffect, useState } from 'react';
import { BrowserRouter, Routes, Route, Navigate, useNavigate, useLocation } from 'react-router-dom';
import { MainApp } from './pages/MainApp';
import { Login } from './pages/Login';
import { Settings } from './pages/Settings';
import { isAuthenticated, getCurrentUser, setAuthTokenFromHash, ensureDemoToken, isDemoSession } from './services/auth';
import { AuthLoadingSkeleton } from './components/AuthLoadingSkeleton';
import { Toaster } from './components/ui/toaster';
import './App.css';

// Protected route wrapper with auth check
function ProtectedRoute({ children }: { children: React.ReactNode }) {
  const navigate = useNavigate();
  const location = useLocation();
  const [isChecking, setIsChecking] = useState(true);
  const [hasToken, setHasToken] = useState<boolean>(isAuthenticated());

  // T110: Check if user is authenticated on mount
  useEffect(() => {
    const checkAuth = async () => {
      // Check for OAuth callback token in URL hash
      const tokenExtracted = setAuthTokenFromHash();
      if (tokenExtracted) {
        console.log('OAuth token extracted from URL hash');
        setHasToken(true);
        setIsChecking(false);
        return;
      }

      if (!isAuthenticated()) {
        const demoReady = await ensureDemoToken();
        if (!demoReady) {
          setHasToken(false);
          setIsChecking(false);
          return;
        }
      }

      setHasToken(true);

      if (isDemoSession()) {
        setIsChecking(false);
        return;
      }

      const token = localStorage.getItem('auth_token');
      // Skip validation for local dev token
      if (token === 'local-dev-token') {
        setIsChecking(false);
        return;
      }

      try {
        // Verify the token is valid by calling getCurrentUser
        await getCurrentUser();
        setIsChecking(false);
      } catch {
        // Token is invalid (401), redirect to login
        console.warn('Authentication failed, redirecting to login');
        localStorage.removeItem('auth_token');
        setHasToken(false);
        setIsChecking(false);
        if (location.pathname !== '/login') {
          navigate('/login', { replace: true, state: { from: location } });
        }
      }
    };

    checkAuth();
  }, [navigate, location]);

  if (isChecking) {
    return <AuthLoadingSkeleton />;
  }

  if (!hasToken) {
    return <Navigate to="/login" replace />;
  }

  return <>{children}</>;
}

function App() {
  return (
    <>
      <BrowserRouter>
        <Routes>
          <Route path="/login" element={<Login />} />
          <Route
            path="/"
            element={
              <ProtectedRoute>
                <MainApp />
              </ProtectedRoute>
            }
          />
          <Route
            path="/settings"
            element={
              <ProtectedRoute>
                <Settings />
              </ProtectedRoute>
            }
          />
        </Routes>
      </BrowserRouter>
      <Toaster />
    </>
  );
}

export default App;
