import { useEffect, useState } from 'react';
import { BrowserRouter, Routes, Route, Navigate, useNavigate, useLocation } from 'react-router-dom';
import { MainApp } from './pages/MainApp';
import { Login } from './pages/Login';
import { Settings } from './pages/Settings';
import { isAuthenticated, getCurrentUser } from './services/auth';
import { AuthLoadingSkeleton } from './components/AuthLoadingSkeleton';
import { Toaster } from './components/ui/toaster';
import './App.css';

// Protected route wrapper with auth check
function ProtectedRoute({ children }: { children: React.ReactNode }) {
  const navigate = useNavigate();
  const location = useLocation();
  const [isChecking, setIsChecking] = useState(true);

  // T110: Check if user is authenticated on mount
  useEffect(() => {
    const checkAuth = async () => {
      if (!isAuthenticated()) {
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
      } catch (err) {
        // Token is invalid (401), redirect to login
        console.warn('Authentication failed, redirecting to login');
        localStorage.removeItem('auth_token');
        navigate('/login', { replace: true, state: { from: location } });
      }
    };

    checkAuth();
  }, [navigate, location]);

  if (!isAuthenticated()) {
    return <Navigate to="/login" replace />;
  }

  if (isChecking) {
    return <AuthLoadingSkeleton />;
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
