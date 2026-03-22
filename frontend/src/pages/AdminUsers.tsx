/**
 * Admin user management panel. Displays all users with role badges
 * and provides approve/block/promote actions. Only accessible to admins.
 */
import { useState, useEffect, useCallback } from 'react';
import { UserCheck, UserX, ShieldCheck, ShieldMinus, Loader2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar';
import { apiFetch, APIException } from '@/services/api';

interface UserRecord {
  user_id: string;
  display_name: string | null;
  avatar_url: string | null;
  role: 'admin' | 'user' | 'pending' | 'blocked';
  approved_by: string | null;
  created_at: string;
  last_login_at: string | null;
}

function roleBadge(role: string) {
  switch (role) {
    case 'admin':
      return <Badge variant="default" className="bg-amber-600 hover:bg-amber-700">Admin</Badge>;
    case 'user':
      return <Badge variant="secondary">User</Badge>;
    case 'pending':
      return <Badge variant="outline" className="text-yellow-500 border-yellow-500">Pending</Badge>;
    case 'blocked':
      return <Badge variant="destructive">Blocked</Badge>;
    default:
      return <Badge variant="outline">{role}</Badge>;
  }
}

function formatDate(iso: string | null): string {
  if (!iso) return '--';
  return new Date(iso).toLocaleDateString(undefined, {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
  });
}

export function AdminUsers() {
  const [users, setUsers] = useState<UserRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [currentUserId, setCurrentUserId] = useState<string | null>(null);
  const [updatingUser, setUpdatingUser] = useState<string | null>(null);

  const fetchUsers = useCallback(async () => {
    try {
      setLoading(true);
      const data = await apiFetch<UserRecord[]>('/api/admin/users');
      setUsers(data);
      setError(null);
    } catch (err) {
      if (err instanceof APIException && err.status === 403) {
        setError('You do not have permission to view user management.');
      } else {
        setError(err instanceof Error ? err.message : 'Failed to load users');
      }
    } finally {
      setLoading(false);
    }
  }, []);

  const fetchCurrentUser = useCallback(async () => {
    try {
      const me = await apiFetch<{ user_id: string }>('/api/me');
      setCurrentUserId(me.user_id);
    } catch {
      // Non-critical -- actions just won't show self-protection
    }
  }, []);

  useEffect(() => {
    fetchUsers();
    fetchCurrentUser();
  }, [fetchUsers, fetchCurrentUser]);

  const updateRole = async (userId: string, role: string) => {
    try {
      setUpdatingUser(userId);
      await apiFetch(`/api/admin/users/${encodeURIComponent(userId)}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ role }),
      });
      await fetchUsers();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to update role');
    } finally {
      setUpdatingUser(null);
    }
  };

  if (loading) {
    return (
      <Card>
        <CardContent className="flex items-center justify-center py-12">
          <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Alert variant="destructive">
        <AlertDescription>{error}</AlertDescription>
      </Alert>
    );
  }

  const isSelf = (userId: string) => userId === currentUserId;

  return (
    <Card>
      <CardHeader>
        <CardTitle>User Management</CardTitle>
        <CardDescription>
          Approve, block, or change roles for registered users.
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="space-y-3">
          {users.length === 0 && (
            <p className="text-sm text-muted-foreground text-center py-6">
              No users registered yet.
            </p>
          )}
          {users.map((u) => {
            const isUpdating = updatingUser === u.user_id;
            const self = isSelf(u.user_id);
            return (
              <div
                key={u.user_id}
                className="flex items-center justify-between gap-4 rounded-lg border p-3"
              >
                <div className="flex items-center gap-3 min-w-0">
                  <Avatar className="h-9 w-9 shrink-0">
                    {u.avatar_url && <AvatarImage src={u.avatar_url} />}
                    <AvatarFallback>
                      {(u.display_name || u.user_id).charAt(0).toUpperCase()}
                    </AvatarFallback>
                  </Avatar>
                  <div className="min-w-0">
                    <p className="text-sm font-medium truncate">
                      {u.display_name || u.user_id}
                      {self && (
                        <span className="ml-2 text-xs text-muted-foreground">(you)</span>
                      )}
                    </p>
                    <p className="text-xs text-muted-foreground truncate">
                      Joined {formatDate(u.created_at)}
                    </p>
                  </div>
                </div>

                <div className="flex items-center gap-2 shrink-0">
                  {roleBadge(u.role)}

                  {/* Approve pending user */}
                  {u.role === 'pending' && (
                    <Button
                      size="sm"
                      variant="outline"
                      disabled={isUpdating}
                      onClick={() => updateRole(u.user_id, 'user')}
                      title="Approve user"
                    >
                      <UserCheck className="h-4 w-4" />
                    </Button>
                  )}

                  {/* Block active user (not self) */}
                  {(u.role === 'user' || u.role === 'pending') && !self && (
                    <Button
                      size="sm"
                      variant="outline"
                      className="text-destructive hover:text-destructive"
                      disabled={isUpdating}
                      onClick={() => updateRole(u.user_id, 'blocked')}
                      title="Block user"
                    >
                      <UserX className="h-4 w-4" />
                    </Button>
                  )}

                  {/* Unblock blocked user */}
                  {u.role === 'blocked' && (
                    <Button
                      size="sm"
                      variant="outline"
                      disabled={isUpdating}
                      onClick={() => updateRole(u.user_id, 'user')}
                      title="Unblock user"
                    >
                      <UserCheck className="h-4 w-4" />
                    </Button>
                  )}

                  {/* Promote to admin (not self, must be user) */}
                  {u.role === 'user' && !self && (
                    <Button
                      size="sm"
                      variant="outline"
                      disabled={isUpdating}
                      onClick={() => updateRole(u.user_id, 'admin')}
                      title="Promote to admin"
                    >
                      <ShieldCheck className="h-4 w-4" />
                    </Button>
                  )}

                  {/* Demote admin to user (not self) */}
                  {u.role === 'admin' && !self && (
                    <Button
                      size="sm"
                      variant="outline"
                      disabled={isUpdating}
                      onClick={() => updateRole(u.user_id, 'user')}
                      title="Demote to user"
                    >
                      <ShieldMinus className="h-4 w-4" />
                    </Button>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </CardContent>
    </Card>
  );
}
