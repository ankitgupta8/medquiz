import { NextRequest, NextResponse } from 'next/server';
import { createSupabaseServerClient } from '@/lib/supabase/server-client';
import { db } from '@/lib/db';

// GET - List all documents for the authenticated user
export async function GET(request: NextRequest) {
  try {
    const supabase = await createSupabaseServerClient();
    const { data: { user }, error: authError } = await supabase.auth.getUser();

    if (authError || !user) {
      return NextResponse.json(
        { success: false, error: 'Authentication required' },
        { status: 401 }
      );
    }

    const userId = user.id;
    const { searchParams } = new URL(request.url);
    const mode = searchParams.get('mode');
    const difficulty = searchParams.get('difficulty');
    const style = searchParams.get('style');
    const search = searchParams.get('search');

    const where: {
      userId: string;
      mode?: string;
      difficulty?: string;
      style?: string;
      OR?: Array<{ title?: { contains: string }; originalText?: { contains: string } }>;
    } = { userId };

    if (mode && ['mcq', 'flashcard'].includes(mode)) {
      where.mode = mode;
    }
    if (difficulty && ['easy', 'medium', 'hard'].includes(difficulty)) {
      where.difficulty = difficulty;
    }
    if (style && ['clinical', 'high_yield', 'exam'].includes(style)) {
      where.style = style;
    }
    if (search) {
      where.OR = [
        { title: { contains: search, mode: 'insensitive' } },
        { originalText: { contains: search, mode: 'insensitive' } },
      ];
    }

    const documents = await db.document.findMany({
      where,
      include: {
        _count: {
          select: { mcqs: true, flashcards: true },
        },
      },
      orderBy: {
        createdAt: 'desc',
      },
    });

    return NextResponse.json({
      success: true,
      data: documents,
    });
  } catch (error) {
    console.error('Error fetching documents:', error);
    return NextResponse.json(
      {
        success: false,
        error: 'Failed to fetch documents',
      },
      { status: 500 }
    );
  }
}

// DELETE - Delete multiple documents (only user's own documents)
export async function DELETE(request: NextRequest) {
  try {
    const supabase = await createSupabaseServerClient();
    const { data: { user }, error: authError } = await supabase.auth.getUser();

    if (authError || !user) {
      return NextResponse.json(
        { success: false, error: 'Authentication required' },
        { status: 401 }
      );
    }

    const userId = user.id;
    const { ids } = await request.json();

    if (!Array.isArray(ids) || ids.length === 0) {
      return NextResponse.json(
        { success: false, error: 'Document IDs are required' },
        { status: 400 }
      );
    }

    // Only delete documents that belong to the user
    const result = await db.document.deleteMany({
      where: {
        id: { in: ids },
        userId, // Ensure user owns these documents
      },
    });

    return NextResponse.json({
      success: true,
      message: `Deleted ${result.count} document(s)`,
    });
  } catch (error) {
    console.error('Error deleting documents:', error);
    return NextResponse.json(
      {
        success: false,
        error: 'Failed to delete documents',
      },
      { status: 500 }
    );
  }
}
