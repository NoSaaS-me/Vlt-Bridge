/**
 * Hugging Face profile metadata attached to a user.
 */
export interface HFProfile {
  username: string;
  name?: string;
  avatar_url?: string;
}

/**
 * User account returned by the backend.
 */
export interface User {
  user_id: string;
  hf_profile?: HFProfile;
  vault_path: string;
  created: string; // ISO 8601 timestamp
}

