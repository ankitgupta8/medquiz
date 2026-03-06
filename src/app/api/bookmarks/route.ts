import { NextRequest, NextResponse } from 'next/server';
import { createSupabaseServerClient } from '@/lib/supabase/server-client';
import { db } from '@/lib/db';

// GET - Fetch all bookmarks for the authenticated user
export async function GET(request: NextRequest) {
  try {
    const supabase = await createSupabaseServerClient();
    const { data: { user }, error: authError } = await supabase.auth.getUser();

    // Return empty array for unauthenticated users
    if (authError || !user) {
      return NextResponse.json({
        success: true,
        data: [],
      });
    }

    const userId = user.id;
    const { searchParams } = new URL(request.url);
    const type = searchParams.get('type'); // 'mcq' or 'flashcard'
    
    const where = { userId, ...(type ? { itemType: type } : {}) };
    
    const bookmarks = await db.bookmark.findMany({
      where,
      orderBy: { createdAt: 'desc' },
    });
    
    // Fetch the actual MCQ or Flashcard data for each bookmark
    const enrichedBookmarks = await Promise.all(
      bookmarks.map(async (bookmark) => {
        try {
          if (bookmark.itemType === 'mcq') {
            const mcq = await db.mCQ.findUnique({
              where: { id: bookmark.itemId },
              include: { document: { select: { title: true, userId: true } } },
            });
            // Only return if the document belongs to the user
            if (mcq && mcq.document.userId === userId) {
              return {
                id: bookmark.id,
                itemType: bookmark.itemType,
                itemId: bookmark.itemId,
                createdAt: bookmark.createdAt.toISOString(),
                item: {
                  id: mcq.id,
                  question: mcq.question,
                  options: { A: mcq.optionA, B: mcq.optionB, C: mcq.optionC, D: mcq.optionD },
                  correctAnswer: mcq.correctAnswer,
                  explanation: mcq.explanation,
                  hint: mcq.hint,
                  difficulty: mcq.difficulty,
                  style: mcq.style,
                  documentTitle: mcq.document?.title || 'Untitled',
                },
              };
            }
          } else if (bookmark.itemType === 'flashcard') {
            const flashcard = await db.flashcard.findUnique({
              where: { id: bookmark.itemId },
              include: { document: { select: { title: true, userId: true } } },
            });
            // Only return if the document belongs to the user
            if (flashcard && flashcard.document.userId === userId) {
              return {
                id: bookmark.id,
                itemType: bookmark.itemType,
                itemId: bookmark.itemId,
                createdAt: bookmark.createdAt.toISOString(),
                item: {
                  id: flashcard.id,
                  front: flashcard.front,
                  back: flashcard.back,
                  mnemonic: flashcard.mnemonic,
                  clinicalCorrelation: flashcard.clinicalCorrelation,
                  keyPoint: flashcard.keyPoint,
                  difficulty: flashcard.difficulty,
                  style: flashcard.style,
                  documentTitle: flashcard.document?.title || 'Untitled',
                },
              };
            }
          }
        } catch (err) {
          console.error(`Error fetching item ${bookmark.itemId}:`, err);
        }
        return null;
      })
    );
    
    // Filter out null values (items that were deleted or don't belong to user)
    const validBookmarks = enrichedBookmarks.filter((b): b is NonNullable<typeof b> => b !== null);
    
    return NextResponse.json({
      success: true,
      data: validBookmarks,
    });
  } catch (error) {
    console.error('Error fetching bookmarks:', error);
    return NextResponse.json(
      { success: false, error: 'Failed to fetch bookmarks' },
      { status: 500 }
    );
  }
}

// POST - Add a new bookmark
export async function POST(request: NextRequest) {
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
    const body = await request.json();
    const { itemType, itemId } = body as {
      itemType: 'mcq' | 'flashcard';
      itemId: string;
    };
    
    // Validate input
    if (!['mcq', 'flashcard'].includes(itemType)) {
      return NextResponse.json(
        { success: false, error: 'itemType must be "mcq" or "flashcard"' },
        { status: 400 }
      );
    }
    
    if (!itemId) {
      return NextResponse.json(
        { success: false, error: 'itemId is required' },
        { status: 400 }
      );
    }
    
    // Verify the item exists and belongs to the user
    if (itemType === 'mcq') {
      const mcq = await db.mCQ.findUnique({
        where: { id: itemId },
        include: { document: { select: { userId: true } } },
      });
      if (!mcq) {
        return NextResponse.json(
          { success: false, error: 'MCQ not found' },
          { status: 404 }
        );
      }
      if (mcq.document.userId !== userId) {
        return NextResponse.json(
          { success: false, error: 'Not authorized to bookmark this item' },
          { status: 403 }
        );
      }
    } else {
      const flashcard = await db.flashcard.findUnique({
        where: { id: itemId },
        include: { document: { select: { userId: true } } },
      });
      if (!flashcard) {
        return NextResponse.json(
          { success: false, error: 'Flashcard not found' },
          { status: 404 }
        );
      }
      if (flashcard.document.userId !== userId) {
        return NextResponse.json(
          { success: false, error: 'Not authorized to bookmark this item' },
          { status: 403 }
        );
      }
    }
    
    // Create bookmark (upsert to handle duplicates)
    const bookmark = await db.bookmark.upsert({
      where: {
        userId_itemType_itemId: { userId, itemType, itemId },
      },
      update: {}, // No update needed if exists
      create: {
        userId,
        itemType,
        itemId,
      },
    });
    
    return NextResponse.json({
      success: true,
      data: bookmark,
      message: 'Bookmark added successfully',
    });
  } catch (error) {
    console.error('Error adding bookmark:', error);
    return NextResponse.json(
      { success: false, error: 'Failed to add bookmark' },
      { status: 500 }
    );
  }
}

// DELETE - Remove a bookmark
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
    const { searchParams } = new URL(request.url);
    const bookmarkId = searchParams.get('id');
    const itemType = searchParams.get('type');
    const itemId = searchParams.get('itemId');
    
    if (bookmarkId) {
      // Delete by bookmark ID (verify ownership)
      const bookmark = await db.bookmark.findUnique({
        where: { id: bookmarkId },
      });
      
      if (!bookmark || bookmark.userId !== userId) {
        return NextResponse.json(
          { success: false, error: 'Bookmark not found or not authorized' },
          { status: 404 }
        );
      }
      
      await db.bookmark.delete({
        where: { id: bookmarkId },
      });
    } else if (itemType && itemId) {
      // Delete by item type and ID
      await db.bookmark.delete({
        where: {
          userId_itemType_itemId: { userId, itemType: itemType as 'mcq' | 'flashcard', itemId },
        },
      });
    } else {
      return NextResponse.json(
        { success: false, error: 'Either bookmark ID or itemType+itemId is required' },
        { status: 400 }
      );
    }
    
    return NextResponse.json({
      success: true,
      message: 'Bookmark removed successfully',
    });
  } catch (error) {
    console.error('Error removing bookmark:', error);
    return NextResponse.json(
      { success: false, error: 'Failed to remove bookmark' },
      { status: 500 }
    );
  }
}
