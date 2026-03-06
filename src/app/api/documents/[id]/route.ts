import { NextRequest, NextResponse } from 'next/server';
import { createSupabaseServerClient } from '@/lib/supabase/server-client';
import { db } from '@/lib/db';

// GET - Get a single document with its MCQs or Flashcards
export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
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
    const { id } = await params;

    const document = await db.document.findUnique({
      where: { id, userId },
      include: {
        mcqs: {
          orderBy: { createdAt: 'asc' },
        },
        flashcards: {
          orderBy: { createdAt: 'asc' },
        },
      },
    });

    if (!document) {
      return NextResponse.json(
        { success: false, error: 'Document not found' },
        { status: 404 }
      );
    }

    return NextResponse.json({
      success: true,
      data: document,
    });
  } catch (error) {
    console.error('Error fetching document:', error);
    return NextResponse.json(
      {
        success: false,
        error: 'Failed to fetch document',
      },
      { status: 500 }
    );
  }
}

// PUT - Update a document
export async function PUT(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
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
    const { id } = await params;
    const body = await request.json();
    const { title, difficulty, style } = body;

    // Check if document belongs to user
    const existingDoc = await db.document.findUnique({
      where: { id },
      select: { userId: true },
    });

    if (!existingDoc || existingDoc.userId !== userId) {
      return NextResponse.json(
        { success: false, error: 'Document not found or not authorized' },
        { status: 404 }
      );
    }

    const updateData: {
      title?: string;
      difficulty?: string;
      style?: string;
    } = {};

    if (title !== undefined) updateData.title = title;
    if (difficulty !== undefined) updateData.difficulty = difficulty;
    if (style !== undefined) updateData.style = style;

    const document = await db.document.update({
      where: { id },
      data: updateData,
    });

    return NextResponse.json({
      success: true,
      data: document,
    });
  } catch (error) {
    console.error('Error updating document:', error);
    return NextResponse.json(
      {
        success: false,
        error: 'Failed to update document',
      },
      { status: 500 }
    );
  }
}

// DELETE - Delete a single document
export async function DELETE(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
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
    const { id } = await params;

    // Check if document belongs to user
    const existingDoc = await db.document.findUnique({
      where: { id },
      select: { userId: true },
    });

    if (!existingDoc || existingDoc.userId !== userId) {
      return NextResponse.json(
        { success: false, error: 'Document not found or not authorized' },
        { status: 404 }
      );
    }

    await db.document.delete({
      where: { id },
    });

    return NextResponse.json({
      success: true,
      message: 'Document deleted successfully',
    });
  } catch (error) {
    console.error('Error deleting document:', error);
    return NextResponse.json(
      {
        success: false,
        error: 'Failed to delete document',
      },
      { status: 500 }
    );
  }
}
