import { useEffect, useState } from 'react';
import { BrowserRouter, Routes, Route, Navigate, useNavigate, useLocation } from 'react-router-dom';
import { MainApp } from './pages/MainApp';
import { Login } from './pages/Login';
import { Settings } from './pages/Settings';
import { SetupWizard } from './pages/SetupWizard';
import { AccountPending } from './pages/AccountPending';
import { AccountBlocked } from './pages/AccountBlocked';
import { isAuthenticated, getCurrentUser, setAuthTokenFromHash } from './services/auth';
import { APIException } from './services/api';
import { AuthLoadingSkeleton } from './components/AuthLoadingSkeleton';
import { Toaster } from './components/ui/toaster';
import { ProjectProvider } from './contexts/ProjectContext';
import { OnboardingProvider, OnboardingOverlay } from './onboarding';
import './App.css';

interface SetupStatus {
  setup_complete: boolean;
  approval_required: boolean;
}

// Protected route wrapper with auth check
function ProtectedRoute({ children, approvalRequired }: { children: React.ReactNode; approvalRequired: boolean }) {
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

      // When approvalRequired is false (open/local mode), bypass server validation
      // for the local dev token so ENABLE_LOCAL_MODE can work in future.
      if (!approvalRequired) {
        const token = localStorage.getItem('auth_token');
        // Skip server-side validation for the local dev token.
        if (token === 'local-dev-token') {
          setHasToken(true);
          setIsChecking(false);
          return;
        }
      }

      if (!isAuthenticated()) {
        setHasToken(false);
        setIsChecking(false);
        return;
      }

      setHasToken(true);

      try {
        // Verify the token is valid by calling getCurrentUser
        await getCurrentUser();
        setIsChecking(false);
      } catch (err) {
        // Handle 403 account status errors (pending/blocked)
        if (err instanceof APIException && err.status === 403) {
          const errorCode = err.detail?.error ?? err.error;
          if (errorCode === 'account_pending') {
            navigate('/pending', { replace: true });
            setIsChecking(false);
            return;
          }
          if (errorCode === 'account_blocked') {
            navigate('/blocked', { replace: true });
            setIsChecking(false);
            return;
          }
        }
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
  const [setupStatus, setSetupStatus] = useState<SetupStatus | null>(null);
  const [setupLoading, setSetupLoading] = useState(true);

  // Check setup status on mount (before auth). Uses raw fetch since
  // this endpoint requires no authentication.
  useEffect(() => {
    fetch('/api/admin/setup-status')
      .then((r) => r.json())
      .then((data: SetupStatus) => {
        setSetupStatus(data);
        setSetupLoading(false);
      })
      .catch(() => {
        // On error assume setup is complete and approval not required
        // (backwards compat / feature disabled)
        setSetupStatus({ setup_complete: true, approval_required: false });
        setSetupLoading(false);
      });
  }, []);

  if (setupLoading) {
    return <AuthLoadingSkeleton />;
  }

  // Show setup wizard when approval is required but no admin exists yet
  const needsSetup =
    setupStatus !== null &&
    setupStatus.approval_required &&
    !setupStatus.setup_complete;

  return (
    <>
      <BrowserRouter>
        <OnboardingProvider>
          <Routes>
            {needsSetup ? (
              <>
                {/* During setup, allow the login callback path so OAuth can complete */}
                <Route path="/login" element={<Login />} />
                <Route path="*" element={<SetupWizard />} />
              </>
            ) : (
              <>
                <Route path="/login" element={<Login />} />
                <Route path="/pending" element={<AccountPending />} />
                <Route path="/blocked" element={<AccountBlocked />} />
                <Route
                  path="/"
                  element={
                    <ProtectedRoute approvalRequired={setupStatus?.approval_required ?? false}>
                      <ProjectProvider>
                        <MainApp />
                      </ProjectProvider>
                    </ProtectedRoute>
                  }
                />
                <Route
                  path="/settings"
                  element={
                    <ProtectedRoute approvalRequired={setupStatus?.approval_required ?? false}>
                      <ProjectProvider>
                        <Settings />
                      </ProjectProvider>
                    </ProtectedRoute>
                  }
                />
              </>
            )}
          </Routes>
          <OnboardingOverlay />
        </OnboardingProvider>
      </BrowserRouter>
      <Toaster />
    </>
  );
}

export default App;
