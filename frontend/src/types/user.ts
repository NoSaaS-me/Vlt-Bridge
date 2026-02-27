/**
 * GitHub profile metadata attached to a user.
 */
export interface GHProfile {
  username: string;
  name?: string;
  avatar_url?: string;
}

/**
 * User account returned by the backend.
 */
export interface User {
  user_id: string;
  gh_profile?: GHProfile;
  vault_path: string;
  created: string; // ISO 8601 timestamp
}

