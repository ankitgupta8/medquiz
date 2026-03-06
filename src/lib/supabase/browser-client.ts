import { createBrowserClient } from '@supabase/auth-helpers-nextjs';

// Fallback configuration for when env vars are not available
const FALLBACK_SUPABASE_URL = "https://fddwplujylgfosenajnu.supabase.co";
const FALLBACK_SUPABASE_ANON_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImZkZHdwbHVqeWxnZm9zZW5ham51Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzI1ODMxNDIsImV4cCI6MjA4ODE1OTE0Mn0.A84IWE9K15_H_NLFDfaLyy67iXk342lnwYB-mdAT4HY";

let configError: string | null = null;
let client: ReturnType<typeof createBrowserClient> | undefined;

export const getConfigError = () => configError;

export const createSupabaseBrowserClient = () => {
  // Return existing client if already created
  if (client) {
    return client;
  }
  
  // Use env vars if available, otherwise use fallback
  const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL || FALLBACK_SUPABASE_URL;
  const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY || FALLBACK_SUPABASE_ANON_KEY;
  
  if (!supabaseUrl || !supabaseAnonKey || supabaseUrl === 'undefined') {
    configError = 'Missing Supabase configuration.';
    console.error(configError);
    throw new Error(configError);
  }
  
  console.log('[Supabase] Creating browser client with URL:', supabaseUrl);
  
  client = createBrowserClient(
    supabaseUrl,
    supabaseAnonKey
  );
  
  return client;
};

// Alias for clarity
export const getSupabaseBrowserClient = createSupabaseBrowserClient;
