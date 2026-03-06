'use client';

import { useState, useEffect, useRef } from 'react';
import { AuthForm } from '@/components/auth/auth-form';
import { Brain, Loader2 } from 'lucide-react';
import type { Session } from '@supabase/supabase-js';
import { createBrowserClient } from '@supabase/auth-helpers-nextjs';

// Hardcoded config
const SUPABASE_URL = "https://fddwplujylgfosenajnu.supabase.co";
const SUPABASE_ANON_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImZkZHdwbHVqeWxnZm9zZW5ham51Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzI1ODMxNDIsImV4cCI6MjA4ODE1OTE0Mn0.A84IWE9K15_H_NLFDfaLyy67iXk342lnwYB-mdAT4HY";

interface AuthWrapperProps {
  children: React.ReactNode;
}

export function AuthWrapper({ children }: AuthWrapperProps) {
  const [session, setSession] = useState<Session | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const initRef = useRef(false);

  useEffect(() => {
    if (initRef.current) return;
    initRef.current = true;

    const supabase = createBrowserClient(SUPABASE_URL, SUPABASE_ANON_KEY);

    console.log('[Auth] Initializing...');

    supabase.auth.getSession().then(({ data, error }) => {
      if (error) {
        console.error('[Auth] Error:', error.message);
        setError(error.message);
        setLoading(false);
        return;
      }
      console.log('[Auth] Session:', data.session?.user?.email || 'No user');
      setSession(data.session);
      setLoading(false);
    });

    const { data: { subscription } } = supabase.auth.onAuthStateChange(
      (event, newSession) => {
        console.log('[Auth] Event:', event);
        if (event === 'SIGNED_IN' || event === 'SIGNED_OUT' || event === 'TOKEN_REFRESHED') {
          setSession(newSession);
        }
      }
    );

    return () => {
      subscription.unsubscribe();
    };
  }, []);

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-background">
        <div className="flex flex-col items-center gap-4">
          <div className="p-3 rounded-xl bg-gradient-to-br from-violet-500 to-purple-600 text-white">
            <Brain className="w-8 h-8" />
          </div>
          <Loader2 className="w-6 h-6 animate-spin text-violet-500" />
          <p className="text-sm text-muted-foreground">Loading...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-background p-4">
        <div className="max-w-md text-center">
          <div className="p-3 rounded-xl bg-red-500 text-white w-fit mx-auto mb-4">
            <Brain className="w-8 h-8" />
          </div>
          <h1 className="text-2xl font-bold mb-2">Error</h1>
          <p className="text-muted-foreground">{error}</p>
          <button 
            onClick={() => window.location.reload()}
            className="mt-4 px-4 py-2 bg-violet-500 text-white rounded-lg hover:bg-violet-600"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  if (!session) {
    return (
      <div className="min-h-screen flex flex-col items-center justify-center bg-background p-4">
        <div className="mb-8 text-center">
          <div className="flex justify-center mb-4">
            <div className="p-3 rounded-xl bg-gradient-to-br from-violet-500 to-purple-600 text-white">
              <Brain className="w-8 h-8" />
            </div>
          </div>
          <h1 className="text-3xl font-bold mb-2">StudyAI</h1>
          <p className="text-muted-foreground">Please sign in to continue</p>
        </div>
        <AuthForm onSuccess={() => window.location.reload()} />
      </div>
    );
  }

  console.log('[Auth] Showing app for:', session.user?.email);
  return <>{children}</>;
}
