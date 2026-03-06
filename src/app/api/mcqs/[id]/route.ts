import { NextRequest, NextResponse } from 'next/server';
import { db } from '@/lib/db';

// GET - Get a single MCQ
export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const { id } = await params;

    const mcq = await db.mCQ.findUnique({
      where: { id },
      include: {
        document: true,
      },
    });

    if (!mcq) {
      return NextResponse.json(
        { success: false, error: 'MCQ not found' },
        { status: 404 }
      );
    }

    return NextResponse.json({
      success: true,
      data: mcq,
    });
  } catch (error) {
    console.error('Error fetching MCQ:', error);
    return NextResponse.json(
      {
        success: false,
        error: 'Failed to fetch MCQ',
      },
      { status: 500 }
    );
  }
}

// PUT - Update an MCQ
export async function PUT(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const { id } = await params;
    const body = await request.json();
    const { question, optionA, optionB, optionC, optionD, correctAnswer, explanation, hint, difficulty, style } = body;

    const updateData: {
      question?: string;
      optionA?: string;
      optionB?: string;
      optionC?: string;
      optionD?: string;
      correctAnswer?: string;
      explanation?: string;
      hint?: string;
      difficulty?: string;
      style?: string;
    } = {};

    if (question !== undefined) updateData.question = question;
    if (optionA !== undefined) updateData.optionA = optionA;
    if (optionB !== undefined) updateData.optionB = optionB;
    if (optionC !== undefined) updateData.optionC = optionC;
    if (optionD !== undefined) updateData.optionD = optionD;
    if (correctAnswer !== undefined) updateData.correctAnswer = correctAnswer;
    if (explanation !== undefined) updateData.explanation = explanation;
    if (hint !== undefined) updateData.hint = hint;
    if (difficulty !== undefined) updateData.difficulty = difficulty;
    if (style !== undefined) updateData.style = style;

    const mcq = await db.mCQ.update({
      where: { id },
      data: updateData,
    });

    return NextResponse.json({
      success: true,
      data: mcq,
    });
  } catch (error) {
    console.error('Error updating MCQ:', error);
    return NextResponse.json(
      {
        success: false,
        error: 'Failed to update MCQ',
      },
      { status: 500 }
    );
  }
}

// DELETE - Delete an MCQ
export async function DELETE(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const { id } = await params;

    await db.mCQ.delete({
      where: { id },
    });

    return NextResponse.json({
      success: true,
      message: 'MCQ deleted successfully',
    });
  } catch (error) {
    console.error('Error deleting MCQ:', error);
    return NextResponse.json(
      {
        success: false,
        error: 'Failed to delete MCQ',
      },
      { status: 500 }
    );
  }
}
