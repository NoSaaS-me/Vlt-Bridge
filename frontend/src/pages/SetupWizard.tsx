/**
 * First-launch setup wizard. Shown when setup_complete is false
 * and approval_required is true. The first user to sign in via
 * GitHub OAuth becomes the system administrator.
 */
import { Github, Shield } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { login } from '@/services/auth';

export function SetupWizard() {
  return (
    <div className="min-h-screen flex items-center justify-center bg-background p-4">
      <Card className="w-full max-w-md">
        <CardHeader className="text-center space-y-2">
          <div className="flex justify-center mb-4">
            <div className="h-16 w-16 rounded-full bg-primary/10 flex items-center justify-center">
              <Shield className="h-8 w-8 text-primary" />
            </div>
          </div>
          <CardTitle className="text-3xl">Welcome to Vlt</CardTitle>
          <CardDescription className="text-base">
            Sign in with GitHub to create your admin account.
            The first account registered becomes the system administrator.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <Button className="w-full" size="lg" onClick={() => login()}>
            <Github className="w-5 h-5 mr-2" />
            Sign in with GitHub
          </Button>
          <p className="text-xs text-center text-muted-foreground mt-4">
            Additional users can be approved from the admin panel after setup.
          </p>
        </CardContent>
      </Card>
    </div>
  );
}
