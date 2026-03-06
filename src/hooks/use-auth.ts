'use client';

import { createSupabaseBrowserClient } from '@/lib/supabase/browser-client';
import { useRouter } from 'next/navigation';
import { useCallback, useState, useEffect } from 'react';
import type { User as SupabaseUser } from '@supabase/supabase-js';

interface User {
  id: string;
  email: string;
  name?: string | null;
  image?: string | null;
}

interface UseAuthReturn {
  user: User | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  login: (email: string, password: string) => Promise<{ success: boolean; error?: string }>;
  register: (email: string, password: string, name?: string) => Promise<{ success: boolean; error?: string }>;
  logout: () => Promise<void>;
}

export function useAuth(): UseAuthReturn {
  const [user, setUser] = useState<User | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isAuthenticating, setIsAuthenticating] = useState(false);
  const router = useRouter();
  const supabase = createSupabaseBrowserClient();

  useEffect(() => {
    // Get initial session
    const getSession = async () => {
      try {
        const { data: { session } } = await supabase.auth.getSession();
        if (session?.user) {
          setUser({
            id: session.user.id,
            email: session.user.email || '',
            name: session.user.user_metadata?.name || null,
            image: session.user.user_metadata?.avatar_url || null,
          });
        }
      } catch (error) {
        console.error('Session error:', error);
      } finally {
        setIsLoading(false);
      }
    };

    getSession();

    // Listen for auth changes
    const { data: { subscription } } = supabase.auth.onAuthStateChange(
      (event, session) => {
        if (session?.user) {
          setUser({
            id: session.user.id,
            email: session.user.email || '',
            name: session.user.user_metadata?.name || null,
            image: session.user.user_metadata?.avatar_url || null,
          });
        } else {
          setUser(null);
        }
        setIsLoading(false);
      }
    );

    return () => {
      subscription.unsubscribe();
    };
  }, [supabase.auth]);

  const login = useCallback(async (email: string, password: string) => {
    setIsAuthenticating(true);
    try {
      const { data, error } = await supabase.auth.signInWithPassword({
        email: email.toLowerCase(),
        password,
      });

      if (error) {
        return { success: false, error: error.message };
      }

      if (data.user) {
        setUser({
          id: data.user.id,
          email: data.user.email || '',
          name: data.user.user_metadata?.name || null,
          image: data.user.user_metadata?.avatar_url || null,
        });
      }

      router.refresh();
      return { success: true };
    } catch (error) {
      console.error('Login error:', error);
      return { success: false, error: 'An unexpected error occurred' };
    } finally {
      setIsAuthenticating(false);
    }
  }, [router, supabase.auth]);

  const register = useCallback(async (email: string, password: string, name?: string) => {
    setIsAuthenticating(true);
    try {
      const { data, error } = await supabase.auth.signUp({
        email: email.toLowerCase(),
        password,
        options: {
          data: {
            name: name || undefined,
          },
        },
      });

      if (error) {
        return { success: false, error: error.message };
      }

      if (data.user) {
        // Check if email confirmation is required
        if (data.session === null) {
          return { success: true, error: 'Please check your email to confirm your account.' };
        }

        setUser({
          id: data.user.id,
          email: data.user.email || '',
          name: data.user.user_metadata?.name || null,
          image: data.user.user_metadata?.avatar_url || null,
        });
      }

      router.refresh();
      return { success: true };
    } catch (error) {
      console.error('Registration error:', error);
      return { success: false, error: 'An unexpected error occurred' };
    } finally {
      setIsAuthenticating(false);
    }
  }, [router, supabase.auth]);

  const logout = useCallback(async () => {
    await supabase.auth.signOut();
    setUser(null);
    router.refresh();
  }, [router, supabase.auth]);

  return {
    user,
    isAuthenticated: !!user,
    isLoading: isLoading || isAuthenticating,
    login,
    register,
    logout,
  };
}
