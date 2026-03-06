import { NextRequest, NextResponse } from 'next/server';
import { db } from '@/lib/db';

// GET - Get a single Flashcard
export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const { id } = await params;

    const flashcard = await db.flashcard.findUnique({
      where: { id },
      include: {
        document: true,
      },
    });

    if (!flashcard) {
      return NextResponse.json(
        { success: false, error: 'Flashcard not found' },
        { status: 404 }
      );
    }

    return NextResponse.json({
      success: true,
      data: flashcard,
    });
  } catch (error) {
    console.error('Error fetching flashcard:', error);
    return NextResponse.json(
      {
        success: false,
        error: 'Failed to fetch flashcard',
      },
      { status: 500 }
    );
  }
}

// PUT - Update a Flashcard
export async function PUT(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const { id } = await params;
    const body = await request.json();
    const { front, back, mnemonic, clinicalCorrelation, keyPoint, difficulty, style } = body;

    const updateData: {
      front?: string;
      back?: string;
      mnemonic?: string | null;
      clinicalCorrelation?: string | null;
      keyPoint?: string | null;
      difficulty?: string;
      style?: string;
    } = {};

    if (front !== undefined) updateData.front = front;
    if (back !== undefined) updateData.back = back;
    if (mnemonic !== undefined) updateData.mnemonic = mnemonic || null;
    if (clinicalCorrelation !== undefined) updateData.clinicalCorrelation = clinicalCorrelation || null;
    if (keyPoint !== undefined) updateData.keyPoint = keyPoint || null;
    if (difficulty !== undefined) updateData.difficulty = difficulty;
    if (style !== undefined) updateData.style = style;

    const flashcard = await db.flashcard.update({
      where: { id },
      data: updateData,
    });

    return NextResponse.json({
      success: true,
      data: flashcard,
    });
  } catch (error) {
    console.error('Error updating flashcard:', error);
    return NextResponse.json(
      {
        success: false,
        error: 'Failed to update flashcard',
      },
      { status: 500 }
    );
  }
}

// DELETE - Delete a Flashcard
export async function DELETE(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const { id } = await params;

    await db.flashcard.delete({
      where: { id },
    });

    return NextResponse.json({
      success: true,
      message: 'Flashcard deleted successfully',
    });
  } catch (error) {
    console.error('Error deleting flashcard:', error);
    return NextResponse.json(
      {
        success: false,
        error: 'Failed to delete flashcard',
      },
      { status: 500 }
    );
  }
}
