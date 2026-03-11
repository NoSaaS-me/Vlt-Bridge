/**
 * Status page shown when a user's account has been blocked by an admin.
 */
import { ShieldX } from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { logout } from '@/services/auth';

export function AccountBlocked() {
  return (
    <div className="min-h-screen flex items-center justify-center bg-background p-4">
      <Card className="w-full max-w-md">
        <CardHeader className="text-center space-y-2">
          <div className="flex justify-center mb-4">
            <div className="h-12 w-12 rounded-full bg-destructive/10 flex items-center justify-center mx-auto">
              <ShieldX className="h-6 w-6 text-destructive" />
            </div>
          </div>
          <CardTitle className="text-2xl">Account Blocked</CardTitle>
          <CardDescription className="text-base">
            Your account has been blocked by an administrator.
            Please contact an admin if you believe this is a mistake.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <Button variant="outline" className="w-full" onClick={() => logout()}>
            Sign Out
          </Button>
        </CardContent>
      </Card>
    </div>
  );
}
