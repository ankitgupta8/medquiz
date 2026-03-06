import { NextResponse } from 'next/server';
import { createSupabaseServerClient } from '@/lib/supabase/server-client';

export async function GET() {
  try {
    const supabase = await createSupabaseServerClient();
    const { data: { user }, error } = await supabase.auth.getUser();

    if (error || !user) {
      return NextResponse.json({
        success: false,
        authenticated: false,
        user: null,
      });
    }

    return NextResponse.json({
      success: true,
      authenticated: true,
      user: {
        id: user.id,
        email: user.email,
        name: user.user_metadata?.name || null,
        image: user.user_metadata?.avatar_url || null,
      },
    });
  } catch (error) {
    console.error('[Auth] Session error:', error);
    return NextResponse.json({
      success: false,
      authenticated: false,
      user: null,
    });
  }
}
