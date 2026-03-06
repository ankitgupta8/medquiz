import { NextRequest, NextResponse } from 'next/server';
import { db } from '@/lib/db';

// GET - List MCQs with filtering
export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const documentId = searchParams.get('documentId');
    const difficulty = searchParams.get('difficulty');
    const style = searchParams.get('style');

    const where: {
      documentId?: string;
      difficulty?: string;
      style?: string;
    } = {};

    if (documentId) where.documentId = documentId;
    if (difficulty && ['easy', 'medium', 'hard'].includes(difficulty)) {
      where.difficulty = difficulty;
    }
    if (style && ['clinical', 'high_yield', 'exam'].includes(style)) {
      where.style = style;
    }

    const mcqs = await db.mCQ.findMany({
      where,
      include: {
        document: {
          select: {
            id: true,
            title: true,
            mode: true,
          },
        },
      },
      orderBy: {
        createdAt: 'desc',
      },
    });

    return NextResponse.json({
      success: true,
      data: mcqs,
    });
  } catch (error) {
    console.error('Error fetching MCQs:', error);
    return NextResponse.json(
      {
        success: false,
        error: 'Failed to fetch MCQs',
      },
      { status: 500 }
    );
  }
}
