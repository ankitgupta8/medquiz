'use client';

import { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import ReactMarkdown from 'react-markdown';
import { 
  BookOpen, 
  Brain, 
  GraduationCap, 
  Layers, 
  Library, 
  Lightbulb, 
  Loader2, 
  Plus, 
  RefreshCw, 
  Sparkles,
  Check,
  X,
  ChevronRight,
  ChevronLeft,
  FileText,
  Clock,
  Target,
  Zap,
  Stethoscope,
  Trash2,
  Eye,
  EyeOff,
  Moon,
  Sun,
  RotateCcw,
  Home as HomeIcon,
  Play,
  AlertCircle,
  Layers3,
  Timer,
  Hash,
  Info,
  Bookmark,
  BookmarkCheck,
  Upload,
  FileUp,
  File,
  ExternalLink,
  User
} from 'lucide-react';
import { UserDropdown } from '@/components/auth/user-dropdown';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { ToggleGroup, ToggleGroupItem } from '@/components/ui/toggle-group';
import { Label } from '@/components/ui/label';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Separator } from '@/components/ui/separator';
import { Input } from '@/components/ui/input';
import { 
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip';
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs';
import { useToast } from '@/hooks/use-toast';
import { useTheme } from 'next-themes';
import { cn } from '@/lib/utils';
import type { Difficulty, Style, Mode, MCQ, Flashcard } from '@/types';

// Types
interface DocumentData {
  id: string;
  title: string | null;
  mode: string;
  difficulty: string;
  style: string;
  createdAt: string;
  _count?: {
    mcqs: number;
    flashcards: number;
  };
}

interface ProcessingStats {
  totalChars: number;
  chunkCount: number;
  avgPerChunk: number;
  estimatedTime: string;
}

// Bookmark types
interface BookmarkItem {
  id: string;
  itemType: 'mcq' | 'flashcard';
  itemId: string;
  createdAt: string;
  item: {
    id: string;
    question?: string;
    front?: string;
    back?: string;
    options?: { A: string; B: string; C: string; D: string };
    correctAnswer?: string;
    explanation?: string;
    hint?: string;
    mnemonic?: string;
    clinicalCorrelation?: string;
    keyPoint?: string;
    difficulty: string;
    style: string;
    documentTitle?: string;
  };
}

// Main Component
export default function StudyAIApp() {
  const { theme, setTheme } = useTheme();
  const { toast } = useToast();
  const [mounted, setMounted] = useState(false);

  // View state
  const [activeTab, setActiveTab] = useState<string>('home');

  // Generation settings
  const [inputText, setInputText] = useState('');
  const [mode, setMode] = useState<Mode>('mcq');
  const [difficulty, setDifficulty] = useState<Difficulty>('medium');
  const [style, setStyle] = useState<Style>('high_yield');
  const [questionCount, setQuestionCount] = useState(10);
  const [questionInputValue, setQuestionInputValue] = useState('10');
  const [selectedModel, setSelectedModel] = useState<string>('gpt-oss');
  const [isGenerating, setIsGenerating] = useState(false);
  const [generationProgress, setGenerationProgress] = useState<string>('');

  // Processing stats
  const [processingStats, setProcessingStats] = useState<ProcessingStats | null>(null);

  // Generated content
  const [currentDocumentId, setCurrentDocumentId] = useState<string | null>(null);
  const [mcqs, setMcqs] = useState<MCQ[]>([]);
  const [flashcards, setFlashcards] = useState<Flashcard[]>([]);

  // Library
  const [documents, setDocuments] = useState<DocumentData[]>([]);

  // Practice mode
  const [practiceIndex, setPracticeIndex] = useState(0);
  const [answers, setAnswers] = useState<Record<string, string>>({});
  const [results, setResults] = useState<Record<string, boolean>>({});
  const [showExplanation, setShowExplanation] = useState(false);
  const [showFlashcardBack, setShowFlashcardBack] = useState(false);

  // Bookmarks
  const [bookmarks, setBookmarks] = useState<BookmarkItem[]>([]);
  const [bookmarkIds, setBookmarkIds] = useState<Set<string>>(new Set());
  const [bookmarkFilter, setBookmarkFilter] = useState<'all' | 'mcq' | 'flashcard'>('all');

  // Practice source tracking
  const [practiceSource, setPracticeSource] = useState<'library' | 'bookmarks'>('library');

  // PDF Upload
  const [inputSource, setInputSource] = useState<'text' | 'pdf'>('text');
  const [pdfFile, setPdfFile] = useState<File | null>(null);
  const [pdfUploadProgress, setPdfUploadProgress] = useState<string>('');
  const [isUploadingPdf, setIsUploadingPdf] = useState(false);

  useEffect(() => {
    setMounted(true);
    fetchDocuments();
    fetchBookmarks();
  }, []);

  // Update processing stats when text or count changes
  useEffect(() => {
    if (inputText.length > 0) {
      updateProcessingStats();
    } else {
      setProcessingStats(null);
    }
  }, [inputText, questionCount]);

  const updateProcessingStats = useCallback(async () => {
    if (inputText.length === 0) return;
    
    try {
      const response = await fetch(
        `/api/generate?textLength=${inputText.length}&questionCount=${questionCount}`
      );
      const data = await response.json();
      if (data.success) {
        setProcessingStats(data.stats);
      }
    } catch (error) {
      console.error('Failed to fetch stats:', error);
    }
  }, [inputText.length, questionCount]);

  const fetchDocuments = async () => {
    try {
      const response = await fetch('/api/documents');
      const data = await response.json();
      if (data.success) {
        setDocuments(data.data);
      }
    } catch (error) {
      console.error('Failed to fetch documents:', error);
    }
  };

  // Bookmark functions
  const fetchBookmarks = async () => {
    try {
      const response = await fetch('/api/bookmarks');
      const data = await response.json();
      if (data.success) {
        setBookmarks(data.data);
        // Create a set of "type-id" for quick lookup
        const ids = new Set(data.data.map((b: BookmarkItem) => `${b.itemType}-${b.itemId}`));
        setBookmarkIds(ids);
      }
    } catch (error) {
      console.error('Failed to fetch bookmarks:', error);
    }
  };

  const isBookmarked = (itemType: 'mcq' | 'flashcard', itemId: string): boolean => {
    return bookmarkIds.has(`${itemType}-${itemId}`);
  };

  const toggleBookmark = async (itemType: 'mcq' | 'flashcard', itemId: string) => {
    try {
      if (isBookmarked(itemType, itemId)) {
        // Remove bookmark
        const response = await fetch(`/api/bookmarks?type=${itemType}&itemId=${itemId}`, {
          method: 'DELETE',
        });
        const data = await response.json();
        if (data.success) {
          setBookmarkIds(prev => {
            const newSet = new Set(prev);
            newSet.delete(`${itemType}-${itemId}`);
            return newSet;
          });
          setBookmarks(prev => prev.filter(b => !(b.itemType === itemType && b.itemId === itemId)));
          toast({
            title: 'Bookmark removed',
            description: 'Item removed from your bookmarks.',
          });
        }
      } else {
        // Add bookmark
        const response = await fetch('/api/bookmarks', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ itemType, itemId }),
        });
        const data = await response.json();
        if (data.success) {
          setBookmarkIds(prev => new Set(prev).add(`${itemType}-${itemId}`));
          // Refresh bookmarks to get the new one with item data
          fetchBookmarks();
          toast({
            title: 'Bookmark added',
            description: 'Item added to your bookmarks.',
          });
        }
      }
    } catch (error) {
      console.error('Error toggling bookmark:', error);
      toast({
        title: 'Error',
        description: 'Failed to update bookmark.',
        variant: 'destructive',
      });
    }
  };

  const removeBookmark = async (bookmarkId: string, itemType: 'mcq' | 'flashcard', itemId: string) => {
    try {
      const response = await fetch(`/api/bookmarks?id=${bookmarkId}`, {
        method: 'DELETE',
      });
      const data = await response.json();
      if (data.success) {
        setBookmarkIds(prev => {
          const newSet = new Set(prev);
          newSet.delete(`${itemType}-${itemId}`);
          return newSet;
        });
        setBookmarks(prev => prev.filter(b => b.id !== bookmarkId));
        toast({
          title: 'Bookmark removed',
          description: 'Item removed from your bookmarks.',
        });
      }
    } catch (error) {
      console.error('Error removing bookmark:', error);
      toast({
        title: 'Error',
        description: 'Failed to remove bookmark.',
        variant: 'destructive',
      });
    }
  };

  const handleGenerate = async () => {
    if (!inputText.trim()) {
      toast({
        title: 'Error',
        description: 'Please enter some text to generate questions.',
        variant: 'destructive',
      });
      return;
    }

    setIsGenerating(true);
    setGenerationProgress('Analyzing text and splitting into chunks...');

    try {
      const response = await fetch('/api/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: inputText,
          mode,
          difficulty,
          style,
          questionCount,
          model: selectedModel,
        }),
      });

      const data = await response.json();

      if (data.success) {
        setCurrentDocumentId(data.documentId || null);
        
        // Transform MCQs from database format
        const transformedMcqs = (data.mcqs || []).map((m: { id: string; question: string; optionA: string; optionB: string; optionC: string; optionD: string; correctAnswer: string; explanation: string; hint: string | null; difficulty: string; style: string }) => ({
          id: m.id,
          question: m.question,
          options: { A: m.optionA, B: m.optionB, C: m.optionC, D: m.optionD },
          correctAnswer: m.correctAnswer as 'A' | 'B' | 'C' | 'D',
          explanation: m.explanation,
          hint: m.hint || undefined,
          difficulty: m.difficulty as Difficulty,
          style: m.style as Style,
        }));
        
        // Transform Flashcards from database format
        const transformedFlashcards = (data.flashcards || []).map((f: { id: string; front: string; back: string; mnemonic: string | null; clinicalCorrelation: string | null; keyPoint: string | null; difficulty: string; style: string }) => ({
          id: f.id,
          front: f.front,
          back: f.back,
          mnemonic: f.mnemonic || undefined,
          clinicalCorrelation: f.clinicalCorrelation || undefined,
          keyPoint: f.keyPoint || undefined,
          difficulty: f.difficulty as Difficulty,
          style: f.style as Style,
        }));
        
        setMcqs(transformedMcqs);
        setFlashcards(transformedFlashcards);
        setActiveTab('results');
        fetchDocuments();
        
        toast({
          title: 'Success!',
          description: `Generated ${transformedMcqs.length || transformedFlashcards.length} ${mode === 'mcq' ? 'questions' : 'flashcards'} from ${data.stats?.totalChunks || 1} chunk(s) in parallel.`,
        });
      } else {
        throw new Error(data.error || 'Generation failed');
      }
    } catch (error) {
      toast({
        title: 'Generation Failed',
        description: error instanceof Error ? error.message : 'Please try again.',
        variant: 'destructive',
      });
    } finally {
      setIsGenerating(false);
      setGenerationProgress('');
    }
  };

  const handlePdfUpload = async () => {
    if (!pdfFile) {
      toast({
        title: 'Error',
        description: 'Please select a PDF file to upload.',
        variant: 'destructive',
      });
      return;
    }

    setIsUploadingPdf(true);
    setPdfUploadProgress('Uploading PDF...');

    try {
      const formData = new FormData();
      formData.append('file', pdfFile);
      formData.append('mode', mode);
      formData.append('difficulty', difficulty);
      formData.append('style', style);
      formData.append('questionCount', questionCount.toString());
      formData.append('model', selectedModel);

      setPdfUploadProgress('Processing PDF pages...');
      
      const response = await fetch('/api/upload-pdf', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (data.success) {
        setCurrentDocumentId(data.documentId || null);
        
        // Transform MCQs from database format
        const transformedMcqs = (data.mcqs || []).map((m: { id: string; question: string; optionA: string; optionB: string; optionC: string; optionD: string; correctAnswer: string; explanation: string; hint: string | null; difficulty: string; style: string }) => ({
          id: m.id,
          question: m.question,
          options: { A: m.optionA, B: m.optionB, C: m.optionC, D: m.optionD },
          correctAnswer: m.correctAnswer as 'A' | 'B' | 'C' | 'D',
          explanation: m.explanation,
          hint: m.hint || undefined,
          difficulty: m.difficulty as Difficulty,
          style: m.style as Style,
        }));
        
        // Transform Flashcards from database format
        const transformedFlashcards = (data.flashcards || []).map((f: { id: string; front: string; back: string; mnemonic: string | null; clinicalCorrelation: string | null; keyPoint: string | null; difficulty: string; style: string }) => ({
          id: f.id,
          front: f.front,
          back: f.back,
          mnemonic: f.mnemonic || undefined,
          clinicalCorrelation: f.clinicalCorrelation || undefined,
          keyPoint: f.keyPoint || undefined,
          difficulty: f.difficulty as Difficulty,
          style: f.style as Style,
        }));
        
        setMcqs(transformedMcqs);
        setFlashcards(transformedFlashcards);
        setActiveTab('results');
        fetchDocuments();
        setPdfFile(null);
        
        toast({
          title: 'Success!',
          description: `Generated ${transformedMcqs.length || transformedFlashcards.length} ${mode === 'mcq' ? 'questions' : 'flashcards'} from ${data.stats?.pdfPages || 0} PDF pages.`,
        });
      } else {
        throw new Error(data.error || 'PDF processing failed');
      }
    } catch (error) {
      toast({
        title: 'PDF Processing Failed',
        description: error instanceof Error ? error.message : 'Please try again.',
        variant: 'destructive',
      });
    } finally {
      setIsUploadingPdf(false);
      setPdfUploadProgress('');
    }
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file && file.type === 'application/pdf') {
      setPdfFile(file);
    } else {
      toast({
        title: 'Invalid File',
        description: 'Please upload a PDF file.',
        variant: 'destructive',
      });
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setPdfFile(file);
    }
  };

  const handleDeleteDocument = async (docId: string) => {
    try {
      await fetch(`/api/documents/${docId}`, { method: 'DELETE' });
      fetchDocuments();
      toast({
        title: 'Deleted',
        description: 'Document removed successfully.',
      });
    } catch (error) {
      toast({
        title: 'Error',
        description: 'Failed to delete document.',
        variant: 'destructive',
      });
    }
  };

  const handleLoadDocument = async (docId: string) => {
    try {
      const response = await fetch(`/api/documents/${docId}`);
      const data = await response.json();
      if (data.success) {
        setCurrentDocumentId(data.data.id);
        
        // Transform MCQs
        const transformedMcqs = (data.data.mcqs || []).map((m: { id: string; question: string; optionA: string; optionB: string; optionC: string; optionD: string; correctAnswer: string; explanation: string; hint: string | null; difficulty: string; style: string }) => ({
          id: m.id,
          question: m.question,
          options: { A: m.optionA, B: m.optionB, C: m.optionC, D: m.optionD },
          correctAnswer: m.correctAnswer as 'A' | 'B' | 'C' | 'D',
          explanation: m.explanation,
          hint: m.hint || undefined,
          difficulty: m.difficulty as Difficulty,
          style: m.style as Style,
        }));
        
        // Transform Flashcards
        const transformedFlashcards = (data.data.flashcards || []).map((f: { id: string; front: string; back: string; mnemonic: string | null; clinicalCorrelation: string | null; keyPoint: string | null; difficulty: string; style: string }) => ({
          id: f.id,
          front: f.front,
          back: f.back,
          mnemonic: f.mnemonic || undefined,
          clinicalCorrelation: f.clinicalCorrelation || undefined,
          keyPoint: f.keyPoint || undefined,
          difficulty: f.difficulty as Difficulty,
          style: f.style as Style,
        }));
        
        setMcqs(transformedMcqs);
        setFlashcards(transformedFlashcards);
        setMode(data.data.mode as Mode);
        setActiveTab('results');
      }
    } catch (error) {
      toast({
        title: 'Error',
        description: 'Failed to load document.',
        variant: 'destructive',
      });
    }
  };

  const startPractice = (practiceMode: 'mcq' | 'flashcard') => {
    if (practiceMode === 'mcq' && mcqs.length === 0) return;
    if (practiceMode === 'flashcard' && flashcards.length === 0) return;
    setPracticeIndex(0);
    setAnswers({});
    setResults({});
    setShowExplanation(false);
    setShowFlashcardBack(false);
    setPracticeSource('library');
    setMode(practiceMode);
    setActiveTab('practice');
  };

  // Start practicing bookmarked items
  const startBookmarkPractice = (practiceMode: 'mcq' | 'flashcard' | 'all') => {
    // Filter bookmarks based on practice mode
    let mcqBookmarks: BookmarkItem[] = [];
    let flashcardBookmarks: BookmarkItem[] = [];
    
    if (practiceMode === 'mcq' || practiceMode === 'all') {
      mcqBookmarks = bookmarks.filter(b => b.itemType === 'mcq');
    }
    if (practiceMode === 'flashcard' || practiceMode === 'all') {
      flashcardBookmarks = bookmarks.filter(b => b.itemType === 'flashcard');
    }
    
    // Convert bookmark items to MCQ/Flashcard format for practice
    const practiceMcqs: MCQ[] = mcqBookmarks.map(b => ({
      id: b.itemId,
      question: b.item.question || '',
      options: b.item.options || { A: '', B: '', C: '', D: '' },
      correctAnswer: (b.item.correctAnswer as 'A' | 'B' | 'C' | 'D') || 'A',
      explanation: b.item.explanation || '',
      hint: b.item.hint,
      difficulty: b.item.difficulty as Difficulty,
      style: b.item.style as Style,
    }));
    
    const practiceFlashcards: Flashcard[] = flashcardBookmarks.map(b => ({
      id: b.itemId,
      front: b.item.front || '',
      back: b.item.back || '',
      mnemonic: b.item.mnemonic,
      clinicalCorrelation: b.item.clinicalCorrelation,
      keyPoint: b.item.keyPoint,
      difficulty: b.item.difficulty as Difficulty,
      style: b.item.style as Style,
    }));
    
    if (practiceMcqs.length === 0 && practiceFlashcards.length === 0) {
      toast({
        title: 'No items to practice',
        description: 'No bookmarked items available for practice.',
        variant: 'destructive',
      });
      return;
    }
    
    // Set the practice data
    setMcqs(practiceMcqs);
    setFlashcards(practiceFlashcards);
    setPracticeIndex(0);
    setAnswers({});
    setResults({});
    setShowExplanation(false);
    setShowFlashcardBack(false);
    setPracticeSource('bookmarks');
    
    // Set mode based on what we're practicing
    if (practiceMode === 'all') {
      // If both types exist, default to mcq, user can switch
      setMode(practiceMcqs.length > 0 ? 'mcq' : 'flashcard');
    } else {
      setMode(practiceMode);
    }
    
    setActiveTab('practice');
  };

  // Filtered bookmarks based on filter state
  const filteredBookmarks = bookmarks.filter(b => {
    if (bookmarkFilter === 'all') return true;
    return b.itemType === bookmarkFilter;
  });

  // Bookmark counts
  const mcqBookmarkCount = bookmarks.filter(b => b.itemType === 'mcq').length;
  const flashcardBookmarkCount = bookmarks.filter(b => b.itemType === 'flashcard').length;


  const handleAnswer = (questionId: string, answer: string) => {
    setAnswers({ ...answers, [questionId]: answer });
  };

  const checkAnswer = (questionId: string, correctAnswer: string) => {
    const isCorrect = answers[questionId] === correctAnswer;
    setResults({ ...results, [questionId]: isCorrect });
    setShowExplanation(true);
  };

  const nextQuestion = () => {
    if (mode === 'mcq' && practiceIndex < mcqs.length - 1) {
      setPracticeIndex(practiceIndex + 1);
      setShowExplanation(false);
    } else if (mode === 'flashcard' && practiceIndex < flashcards.length - 1) {
      setPracticeIndex(practiceIndex + 1);
      setShowFlashcardBack(false);
    }
  };

  const prevQuestion = () => {
    if (practiceIndex > 0) {
      setPracticeIndex(practiceIndex - 1);
      setShowExplanation(false);
      setShowFlashcardBack(false);
    }
  };

  const calculateScore = () => {
    const correct = Object.values(results).filter(Boolean).length;
    return { correct, total: mcqs.length };
  };

  // Difficulty color mapping
  const difficultyColors = {
    easy: 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-100',
    medium: 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-100',
    hard: 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-100',
  };

  // Style icon mapping
  const styleIcons = {
    clinical: Stethoscope,
    high_yield: Zap,
    exam: Target,
  };

  if (!mounted) return null;

  return (
    <TooltipProvider>
      <div className="min-h-screen flex flex-col bg-gradient-to-br from-slate-50 via-white to-slate-100 dark:from-slate-950 dark:via-slate-900 dark:to-slate-950">
        {/* Header */}
        <header className="sticky top-0 z-50 border-b bg-white/80 dark:bg-slate-900/80 backdrop-blur-sm">
          <div className="container mx-auto px-4 py-3 flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-xl bg-gradient-to-br from-violet-500 to-purple-600 text-white">
                <Brain className="w-6 h-6" />
              </div>
              <div>
                <h1 className="text-xl font-bold bg-gradient-to-r from-violet-600 to-purple-600 bg-clip-text text-transparent">
                  StudyAI
                </h1>
                <p className="text-xs text-muted-foreground">MCQ & Flashcard Generator</p>
              </div>
            </div>
            
            <nav className="flex items-center gap-2">
              <Button
                variant={activeTab === 'home' ? 'default' : 'ghost'}
                size="sm"
                onClick={() => setActiveTab('home')}
                className="gap-2"
              >
                <HomeIcon className="w-4 h-4" />
                <span className="hidden sm:inline">Generate</span>
              </Button>
              <Button
                variant={activeTab === 'results' ? 'default' : 'ghost'}
                size="sm"
                onClick={() => setActiveTab('results')}
                disabled={mcqs.length === 0 && flashcards.length === 0}
                className="gap-2"
              >
                <FileText className="w-4 h-4" />
                <span className="hidden sm:inline">Results</span>
              </Button>
              <Button
                variant={activeTab === 'library' ? 'default' : 'ghost'}
                size="sm"
                onClick={() => setActiveTab('library')}
                className="gap-2"
              >
                <Library className="w-4 h-4" />
                <span className="hidden sm:inline">Library</span>
              </Button>
              <Button
                variant={activeTab === 'bookmarks' ? 'default' : 'ghost'}
                size="sm"
                onClick={() => setActiveTab('bookmarks')}
                className="gap-2"
              >
                <Bookmark className="w-4 h-4" />
                <span className="hidden sm:inline">Bookmarks</span>
                {bookmarks.length > 0 && (
                  <Badge variant="secondary" className="ml-1 h-5 px-1.5 text-xs">
                    {bookmarks.length}
                  </Badge>
                )}
              </Button>
              
              <Separator orientation="vertical" className="h-8 mx-2" />
              
              <Button
                variant="ghost"
                size="icon"
                onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
              >
                {theme === 'dark' ? (
                  <Sun className="w-5 h-5" />
                ) : (
                  <Moon className="w-5 h-5" />
                )}
              </Button>
              
              <Separator orientation="vertical" className="h-8 mx-2" />
              
              <UserDropdown />
            </nav>
          </div>
        </header>

        {/* Main Content */}
        <main className="flex-1 container mx-auto px-4 py-6">
          <AnimatePresence mode="wait">
            {/* Home Tab */}
            {activeTab === 'home' && (
              <motion.div
                key="home"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                className="max-w-4xl mx-auto"
              >
                <Card className="border-2 border-dashed">
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Sparkles className="w-5 h-5 text-violet-500" />
                      Generate Learning Content
                    </CardTitle>
                    <CardDescription>
                      Paste your OCR or academic text and choose your preferences. Large texts are automatically split into chunks and processed in parallel.
                    </CardDescription>
                    <a
                      href="https://console.mistral.ai/build/document-ai/ocr-playground"
                      target="_blank"
                      rel="noopener noreferrer"
                      className="inline-flex items-center gap-1.5 text-sm text-violet-600 hover:text-violet-700 dark:text-violet-400 dark:hover:text-violet-300 mt-1"
                    >
                      <FileText className="w-4 h-4" />
                      Extract text from PDFs with Mistral OCR
                      <ExternalLink className="w-3.5 h-3.5" />
                    </a>
                  </CardHeader>
                  
                  <CardContent className="space-y-6">
                    {/* Input Source Selection */}
                    <div className="space-y-3">
                      <Label className="text-base font-medium">Input Source</Label>
                      <ToggleGroup
                        type="single"
                        value={inputSource}
                        onValueChange={(value) => value && setInputSource(value as 'text' | 'pdf')}
                        className="justify-start gap-2"
                      >
                        <ToggleGroupItem
                          value="text"
                          className="gap-2 px-6 data-[state=on]:bg-violet-100 data-[state=on]:text-violet-900 dark:data-[state=on]:bg-violet-900 dark:data-[state=on]:text-violet-100"
                        >
                          <FileText className="w-4 h-4" />
                          Paste Text
                        </ToggleGroupItem>
                        <ToggleGroupItem
                          value="pdf"
                          className="gap-2 px-6 data-[state=on]:bg-violet-100 data-[state=on]:text-violet-900 dark:data-[state=on]:bg-violet-900 dark:data-[state=on]:text-violet-100"
                        >
                          <FileUp className="w-4 h-4" />
                          Upload PDF
                        </ToggleGroupItem>
                      </ToggleGroup>
                    </div>

                    {/* Text Input */}
                    {inputSource === 'text' && (
                      <div className="space-y-2">
                        <Label htmlFor="text-input" className="text-base font-medium">
                          Academic Text
                        </Label>
                        <Textarea
                          id="text-input"
                          placeholder="Paste OCR or academic text here... Large texts will be automatically split into 10,000 character chunks and processed in parallel."
                          value={inputText}
                          onChange={(e) => setInputText(e.target.value)}
                          className="min-h-[200px] resize-none"
                        />
                        <div className="flex items-center justify-between">
                          <p className="text-xs text-muted-foreground">
                            {inputText.length.toLocaleString()} characters
                            {inputText.length > 10000 && (
                              <span className="ml-2 text-violet-600 dark:text-violet-400">
                                (~{Math.ceil(inputText.length / 10000)} chunks)
                              </span>
                            )}
                          </p>
                          {processingStats && (
                            <Badge variant="outline" className="gap-1">
                              <Layers3 className="w-3 h-3" />
                              {processingStats.chunkCount} chunk{processingStats.chunkCount > 1 ? 's' : ''}
                            </Badge>
                          )}
                        </div>
                      </div>
                    )}

                    {/* PDF Upload */}
                    {inputSource === 'pdf' && (
                      <div className="space-y-4">
                        <Label className="text-base font-medium">PDF Document</Label>
                        <div
                          className={cn(
                            "border-2 border-dashed rounded-lg p-8 text-center transition-colors cursor-pointer",
                            "hover:border-violet-400 hover:bg-violet-50/50 dark:hover:bg-violet-950/50",
                            pdfFile && "border-violet-500 bg-violet-50/50 dark:bg-violet-950/50"
                          )}
                          onDrop={handleDrop}
                          onDragOver={(e) => e.preventDefault()}
                          onClick={() => document.getElementById('pdf-input')?.click()}
                        >
                          <input
                            id="pdf-input"
                            type="file"
                            accept=".pdf"
                            className="hidden"
                            onChange={handleFileSelect}
                          />
                          {pdfFile ? (
                            <div className="space-y-2">
                              <File className="w-12 h-12 mx-auto text-violet-500" />
                              <p className="font-medium text-violet-900 dark:text-violet-100">
                                {pdfFile.name}
                              </p>
                              <p className="text-sm text-muted-foreground">
                                {(pdfFile.size / 1024 / 1024).toFixed(2)} MB
                              </p>
                              <Button
                                variant="outline"
                                size="sm"
                                className="mt-2"
                                onClick={(e) => {
                                  e.stopPropagation();
                                  setPdfFile(null);
                                }}
                              >
                                Remove
                              </Button>
                            </div>
                          ) : (
                            <div className="space-y-2">
                              <Upload className="w-12 h-12 mx-auto text-muted-foreground" />
                              <p className="font-medium">Drop your PDF here</p>
                              <p className="text-sm text-muted-foreground">
                                or click to browse (any size - processed in chunks)
                              </p>
                            </div>
                          )}
                        </div>
                        <p className="text-xs text-muted-foreground">
                          <strong>How it works:</strong> PDFs are split into 30-page chunks and processed in parallel with Mistral OCR, 
                          then content is generated using your selected AI model.
                        </p>
                      </div>
                    )}

                    {/* Processing Stats Card */}
                    {processingStats && processingStats.chunkCount > 0 && (
                      <Card className="bg-violet-50 dark:bg-violet-950/50 border-violet-200 dark:border-violet-800">
                        <CardContent className="p-4">
                          <div className="flex items-center gap-2 mb-2">
                            <Info className="w-4 h-4 text-violet-600" />
                            <span className="font-medium text-violet-900 dark:text-violet-100">
                              Processing Preview
                            </span>
                          </div>
                          <div className="grid grid-cols-3 gap-4 text-sm">
                            <div>
                              <p className="text-muted-foreground">Total Chunks</p>
                              <p className="font-semibold">{processingStats.chunkCount}</p>
                            </div>
                            <div>
                              <p className="text-muted-foreground">~Per Chunk</p>
                              <p className="font-semibold">~{processingStats.avgPerChunk} items</p>
                            </div>
                            <div>
                              <p className="text-muted-foreground">Est. Time</p>
                              <p className="font-semibold flex items-center gap-1">
                                <Timer className="w-3 h-3" />
                                {processingStats.estimatedTime}
                              </p>
                            </div>
                          </div>
                        </CardContent>
                      </Card>
                    )}

                    {/* AI Model Selection */}
                    <div className="space-y-3">
                      <Label className="text-base font-medium">AI Model</Label>
                      <Select value={selectedModel} onValueChange={setSelectedModel}>
                        <SelectTrigger className="w-full">
                          <SelectValue placeholder="Select AI model" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="gpt-oss">
                            <div className="flex items-center gap-2">
                              <Brain className="w-4 h-4" />
                              <span>GPT OSS 120B</span>
                              <span className="text-xs text-muted-foreground ml-auto">Default</span>
                            </div>
                          </SelectItem>
                          <SelectItem value="deepseek">
                            <div className="flex items-center gap-2">
                              <Zap className="w-4 h-4" />
                              <span>DeepSeek V3 0324</span>
                            </div>
                          </SelectItem>
                          <SelectItem value="glm">
                            <div className="flex items-center gap-2">
                              <Sparkles className="w-4 h-4" />
                              <span>GLM 4.7</span>
                            </div>
                          </SelectItem>
                          <SelectItem value="kimi">
                            <div className="flex items-center gap-2">
                              <Lightbulb className="w-4 h-4" />
                              <span>Kimi K2</span>
                            </div>
                          </SelectItem>
                        </SelectContent>
                      </Select>
                      <p className="text-xs text-muted-foreground">
                        Choose the AI model for generating your content. Each model has different strengths.
                      </p>
                    </div>

                    {/* Mode Selection */}
                    <div className="space-y-3">
                      <Label className="text-base font-medium">Content Type</Label>
                      <ToggleGroup
                        type="single"
                        value={mode}
                        onValueChange={(value) => value && setMode(value as Mode)}
                        className="justify-start gap-2"
                      >
                        <ToggleGroupItem
                          value="mcq"
                          className="gap-2 px-6 data-[state=on]:bg-violet-100 data-[state=on]:text-violet-900 dark:data-[state=on]:bg-violet-900 dark:data-[state=on]:text-violet-100"
                        >
                          <GraduationCap className="w-4 h-4" />
                          MCQ Questions
                        </ToggleGroupItem>
                        <ToggleGroupItem
                          value="flashcard"
                          className="gap-2 px-6 data-[state=on]:bg-violet-100 data-[state=on]:text-violet-900 dark:data-[state=on]:bg-violet-900 dark:data-[state=on]:text-violet-100"
                        >
                          <Layers className="w-4 h-4" />
                          Flashcards
                        </ToggleGroupItem>
                      </ToggleGroup>
                    </div>

                    {/* Question Count */}
                    <div className="space-y-3">
                      <div className="flex items-center gap-2">
                        <Label className="text-base font-medium">Number of {mode === 'mcq' ? 'Questions' : 'Cards'}</Label>
                        <Tooltip>
                          <TooltipTrigger>
                            <Info className="w-4 h-4 text-muted-foreground" />
                          </TooltipTrigger>
                          <TooltipContent>
                            <p className="max-w-xs">Questions are distributed proportionally across chunks based on text length. Each chunk is processed in parallel.</p>
                          </TooltipContent>
                        </Tooltip>
                      </div>
                      <div className="flex items-center gap-4">
                        <Input
                          type="text"
                          inputMode="numeric"
                          pattern="[0-9]*"
                          value={questionInputValue}
                          onChange={(e) => {
                            const val = e.target.value.replace(/[^0-9]/g, '');
                            setQuestionInputValue(val);
                            const parsed = parseInt(val);
                            if (!isNaN(parsed) && parsed >= 1) {
                              setQuestionCount(parsed);
                            }
                          }}
                          onBlur={() => {
                            const parsed = parseInt(questionInputValue);
                            if (isNaN(parsed) || parsed < 1) {
                              setQuestionCount(10);
                              setQuestionInputValue('10');
                            }
                          }}
                          className="w-20 text-center font-semibold text-foreground bg-background"
                          placeholder="10"
                        />
                        <span className="text-sm text-muted-foreground">
                          {mode === 'mcq' ? 'questions' : 'cards'}
                        </span>
                        <div className="flex gap-1 flex-wrap">
                          {[10, 20, 30, 50, 100].map((n) => (
                            <Button
                              key={n}
                              variant={questionCount === n ? 'default' : 'outline'}
                              size="sm"
                              onClick={() => {
                                setQuestionCount(n);
                                setQuestionInputValue(String(n));
                              }}
                            >
                              {n}
                            </Button>
                          ))}
                        </div>
                      </div>
                    </div>

                    {/* Difficulty Selection */}
                    <div className="space-y-3">
                      <Label className="text-base font-medium">Difficulty Level</Label>
                      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                        {(['easy', 'medium', 'hard'] as const).map((d) => (
                          <Card
                            key={d}
                            className={cn(
                              'cursor-pointer transition-all hover:border-violet-300',
                              difficulty === d && 'border-violet-500 ring-2 ring-violet-200 dark:ring-violet-800'
                            )}
                            onClick={() => setDifficulty(d)}
                          >
                            <CardContent className="p-4">
                              <div className="flex items-center gap-2 mb-2">
                                <Badge className={difficultyColors[d]}>
                                  {d.charAt(0).toUpperCase() + d.slice(1)}
                                </Badge>
                              </div>
                              <p className="text-sm text-muted-foreground">
                                {d === 'easy' && 'Recall-based questions testing basic knowledge'}
                                {d === 'medium' && 'Concept integration requiring understanding'}
                                {d === 'hard' && 'Application and multi-step reasoning'}
                              </p>
                            </CardContent>
                          </Card>
                        ))}
                      </div>
                    </div>

                    {/* Style Selection */}
                    <div className="space-y-3">
                      <Label className="text-base font-medium">Question Style</Label>
                      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                        {(['clinical', 'high_yield', 'exam'] as const).map((s) => {
                          const Icon = styleIcons[s];
                          return (
                            <Card
                              key={s}
                              className={cn(
                                'cursor-pointer transition-all hover:border-violet-300',
                                style === s && 'border-violet-500 ring-2 ring-violet-200 dark:ring-violet-800'
                              )}
                              onClick={() => setStyle(s)}
                            >
                              <CardContent className="p-4">
                                <div className="flex items-center gap-2 mb-2">
                                  <Icon className="w-5 h-5 text-violet-500" />
                                  <span className="font-medium">
                                    {s === 'high_yield' ? 'High-Yield' : s.charAt(0).toUpperCase() + s.slice(1)}
                                  </span>
                                </div>
                                <p className="text-sm text-muted-foreground">
                                  {s === 'clinical' && 'Include patient scenarios'}
                                  {s === 'high_yield' && 'Direct fact-based'}
                                  {s === 'exam' && 'Pattern recognition'}
                                </p>
                              </CardContent>
                            </Card>
                          );
                        })}
                      </div>
                    </div>
                  </CardContent>

                  <CardFooter className="flex-col gap-3">
                    {inputSource === 'text' ? (
                      <Button
                        size="lg"
                        className="w-full gap-2 bg-gradient-to-r from-violet-600 to-purple-600 hover:from-violet-700 hover:to-purple-700"
                        onClick={handleGenerate}
                        disabled={isGenerating || !inputText.trim()}
                      >
                        {isGenerating ? (
                          <>
                            <Loader2 className="w-5 h-5 animate-spin" />
                            {generationProgress || 'Generating...'}
                          </>
                        ) : (
                          <>
                            <Sparkles className="w-5 h-5" />
                            Generate {questionCount} {mode === 'mcq' ? 'MCQs' : 'Flashcards'}
                          </>
                        )}
                      </Button>
                    ) : (
                      <Button
                        size="lg"
                        className="w-full gap-2 bg-gradient-to-r from-violet-600 to-purple-600 hover:from-violet-700 hover:to-purple-700"
                        onClick={handlePdfUpload}
                        disabled={isUploadingPdf || !pdfFile}
                      >
                        {isUploadingPdf ? (
                          <>
                            <Loader2 className="w-5 h-5 animate-spin" />
                            {pdfUploadProgress || 'Processing...'}
                          </>
                        ) : (
                          <>
                            <Upload className="w-5 h-5" />
                            Process PDF & Generate {questionCount} {mode === 'mcq' ? 'MCQs' : 'Flashcards'}
                          </>
                        )}
                      </Button>
                    )}
                    {isGenerating && processingStats && (
                      <p className="text-sm text-muted-foreground text-center">
                        Processing {processingStats.chunkCount} chunk{processingStats.chunkCount > 1 ? 's' : ''} in parallel...
                        This may take ~{processingStats.estimatedTime}.
                      </p>
                    )}
                    {isUploadingPdf && (
                      <p className="text-sm text-muted-foreground text-center">
                        Processing PDF... This may take a few minutes depending on the number of pages.
                      </p>
                    )}
                  </CardFooter>
                </Card>
              </motion.div>
            )}

            {/* Results Tab */}
            {activeTab === 'results' && (
              <motion.div
                key="results"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                className="max-w-4xl mx-auto"
              >
                <div className="flex items-center justify-between mb-6">
                  <div>
                    <h2 className="text-2xl font-bold">Generated Content</h2>
                    <p className="text-muted-foreground">
                      {mcqs.length || flashcards.length} {mode === 'mcq' ? 'questions' : 'flashcards'} generated
                    </p>
                  </div>
                  <div className="flex gap-2">
                    <Button
                      variant="outline"
                      onClick={() => startPractice(mode)}
                      disabled={mode === 'mcq' ? mcqs.length === 0 : flashcards.length === 0}
                      className="gap-2"
                    >
                      <Play className="w-4 h-4" />
                      Practice
                    </Button>
                    <Button
                      variant="outline"
                      onClick={() => setActiveTab('home')}
                      className="gap-2"
                    >
                      <Plus className="w-4 h-4" />
                      New
                    </Button>
                  </div>
                </div>

                {/* MCQ List */}
                {mode === 'mcq' && mcqs.length > 0 && (
                  <ScrollArea className="h-[calc(100vh-280px)]">
                    <div className="space-y-4 pr-4">
                      {mcqs.map((mcq, index) => (
                        <Card key={mcq.id} className="overflow-hidden">
                          <CardHeader className="pb-3">
                            <div className="flex items-start justify-between">
                              <div className="flex items-center gap-2">
                                <Badge variant="outline">Q{index + 1}</Badge>
                                <Badge className={difficultyColors[mcq.difficulty]}>
                                  {mcq.difficulty}
                                </Badge>
                                <Badge variant="secondary">
                                  {mcq.style === 'high_yield' ? 'High-Yield' : mcq.style}
                                </Badge>
                              </div>
                              <TooltipProvider>
                                <Tooltip>
                                  <TooltipTrigger asChild>
                                    <Button
                                      variant="ghost"
                                      size="icon"
                                      onClick={() => toggleBookmark('mcq', mcq.id)}
                                      className={cn(
                                        isBookmarked('mcq', mcq.id)
                                          ? 'text-violet-500 hover:text-violet-700 hover:bg-violet-50 dark:hover:bg-violet-950'
                                          : 'hover:bg-slate-50 dark:hover:bg-slate-800'
                                      )}
                                    >
                                      {isBookmarked('mcq', mcq.id) ? (
                                        <BookmarkCheck className="w-5 h-5" />
                                      ) : (
                                        <Bookmark className="w-5 h-5" />
                                      )}
                                    </Button>
                                  </TooltipTrigger>
                                  <TooltipContent>
                                    {isBookmarked('mcq', mcq.id) ? 'Remove bookmark' : 'Add bookmark'}
                                  </TooltipContent>
                                </Tooltip>
                              </TooltipProvider>
                            </div>
                            <CardTitle className="text-lg mt-2">{mcq.question}</CardTitle>
                          </CardHeader>
                          <CardContent>
                            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                              {(['A', 'B', 'C', 'D'] as const).map((letter) => (
                                <div
                                  key={letter}
                                  className={cn(
                                    'p-3 rounded-lg border transition-colors',
                                    letter === mcq.correctAnswer
                                      ? 'border-green-500 bg-green-50 dark:bg-green-950'
                                      : 'hover:bg-slate-50 dark:hover:bg-slate-800'
                                  )}
                                >
                                  <span className="font-medium">{letter}.</span> {mcq.options[letter]}
                                  {letter === mcq.correctAnswer && (
                                    <Check className="inline-block ml-2 w-4 h-4 text-green-500" />
                                  )}
                                </div>
                              ))}
                            </div>
                            <div className="mt-4 p-3 bg-slate-50 dark:bg-slate-800 rounded-lg">
                              <p className="text-sm font-medium mb-1">Explanation:</p>
                              <p className="text-sm text-muted-foreground">{mcq.explanation}</p>
                            </div>
                            {mcq.hint && (
                              <div className="mt-2 p-3 bg-amber-50 dark:bg-amber-950 rounded-lg">
                                <p className="text-sm font-medium mb-1 flex items-center gap-1">
                                  <Lightbulb className="w-4 h-4 text-amber-500" />
                                  Hint:
                                </p>
                                <p className="text-sm text-muted-foreground">{mcq.hint}</p>
                              </div>
                            )}
                          </CardContent>
                        </Card>
                      ))}
                    </div>
                  </ScrollArea>
                )}

                {/* Flashcard List */}
                {mode === 'flashcard' && flashcards.length > 0 && (
                  <ScrollArea className="h-[calc(100vh-280px)]">
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 pr-4">
                      {flashcards.map((card, index) => (
                        <Card key={card.id} className="overflow-hidden">
                          <CardHeader className="pb-2">
                            <div className="flex items-center justify-between">
                              <div className="flex items-center gap-2">
                                <Badge variant="outline">#{index + 1}</Badge>
                                <Badge className={difficultyColors[card.difficulty]}>
                                  {card.difficulty}
                                </Badge>
                              </div>
                              <TooltipProvider>
                                <Tooltip>
                                  <TooltipTrigger asChild>
                                    <Button
                                      variant="ghost"
                                      size="icon"
                                      onClick={() => toggleBookmark('flashcard', card.id)}
                                      className={cn(
                                        isBookmarked('flashcard', card.id)
                                          ? 'text-violet-500 hover:text-violet-700 hover:bg-violet-50 dark:hover:bg-violet-950'
                                          : 'hover:bg-slate-50 dark:hover:bg-slate-800'
                                      )}
                                    >
                                      {isBookmarked('flashcard', card.id) ? (
                                        <BookmarkCheck className="w-5 h-5" />
                                      ) : (
                                        <Bookmark className="w-5 h-5" />
                                      )}
                                    </Button>
                                  </TooltipTrigger>
                                  <TooltipContent>
                                    {isBookmarked('flashcard', card.id) ? 'Remove bookmark' : 'Add bookmark'}
                                  </TooltipContent>
                                </Tooltip>
                              </TooltipProvider>
                            </div>
                          </CardHeader>
                          <CardContent className="space-y-3">
                            <div className="p-3 bg-violet-50 dark:bg-violet-950 rounded-lg">
                              <p className="text-xs font-medium text-violet-600 dark:text-violet-400 mb-1">Front</p>
                              <p className="font-medium">{card.front}</p>
                            </div>
                            <div className="p-3 bg-emerald-50 dark:bg-emerald-950 rounded-lg">
                              <p className="text-xs font-medium text-emerald-600 dark:text-emerald-400 mb-1">Back</p>
                              <div className="prose prose-sm dark:prose-invert max-w-none flashcard-markdown">
                                <ReactMarkdown>{card.back}</ReactMarkdown>
                              </div>
                            </div>
                            {card.mnemonic && (
                              <div className="p-2 bg-amber-50 dark:bg-amber-950 rounded-lg">
                                <p className="text-xs font-medium text-amber-600 dark:text-amber-400 mb-1">
                                  <Lightbulb className="w-3 h-3 inline mr-1" />Mnemonic
                                </p>
                                <div className="prose prose-sm dark:prose-invert max-w-none flashcard-markdown">
                                  <ReactMarkdown>{card.mnemonic}</ReactMarkdown>
                                </div>
                              </div>
                            )}
                            {card.clinicalCorrelation && (
                              <div className="p-2 bg-blue-50 dark:bg-blue-950 rounded-lg">
                                <p className="text-xs font-medium text-blue-600 dark:text-blue-400 mb-1">
                                  <Stethoscope className="w-3 h-3 inline mr-1" />Clinical
                                </p>
                                <div className="prose prose-sm dark:prose-invert max-w-none flashcard-markdown">
                                  <ReactMarkdown>{card.clinicalCorrelation}</ReactMarkdown>
                                </div>
                              </div>
                            )}
                          </CardContent>
                        </Card>
                      ))}
                    </div>
                  </ScrollArea>
                )}
              </motion.div>
            )}

            {/* Library Tab */}
            {activeTab === 'library' && (
              <motion.div
                key="library"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                className="max-w-4xl mx-auto"
              >
                <div className="flex items-center justify-between mb-6">
                  <div>
                    <h2 className="text-2xl font-bold">Your Library</h2>
                    <p className="text-muted-foreground">{documents.length} documents saved</p>
                  </div>
                  <Button variant="outline" onClick={fetchDocuments} className="gap-2">
                    <RefreshCw className="w-4 h-4" />
                    Refresh
                  </Button>
                </div>

                {documents.length === 0 ? (
                  <Card className="border-dashed">
                    <CardContent className="flex flex-col items-center justify-center py-12">
                      <Library className="w-12 h-12 text-muted-foreground mb-4" />
                      <h3 className="text-lg font-medium mb-2">No documents yet</h3>
                      <p className="text-muted-foreground text-center mb-4">
                        Generate your first MCQs or Flashcards to see them here.
                      </p>
                      <Button onClick={() => setActiveTab('home')} className="gap-2">
                        <Plus className="w-4 h-4" />
                        Generate Content
                      </Button>
                    </CardContent>
                  </Card>
                ) : (
                  <ScrollArea className="h-[calc(100vh-280px)]">
                    <div className="space-y-3 pr-4">
                      {documents.map((doc) => (
                        <Card
                          key={doc.id}
                          className="hover:border-violet-300 transition-colors cursor-pointer"
                          onClick={() => handleLoadDocument(doc.id)}
                        >
                          <CardContent className="p-4">
                            <div className="flex items-center justify-between">
                              <div className="flex items-center gap-3">
                                <div className={cn(
                                  'p-2 rounded-lg',
                                  doc.mode === 'mcq' ? 'bg-violet-100 dark:bg-violet-900' : 'bg-emerald-100 dark:bg-emerald-900'
                                )}>
                                  {doc.mode === 'mcq' ? (
                                    <GraduationCap className="w-5 h-5 text-violet-600 dark:text-violet-300" />
                                  ) : (
                                    <Layers className="w-5 h-5 text-emerald-600 dark:text-emerald-300" />
                                  )}
                                </div>
                                <div>
                                  <h3 className="font-medium">{doc.title || 'Untitled'}</h3>
                                  <div className="flex items-center gap-2 mt-1">
                                    <Badge variant="outline" className={difficultyColors[doc.difficulty as Difficulty]}>
                                      {doc.difficulty}
                                    </Badge>
                                    <Badge variant="secondary">
                                      {doc.style === 'high_yield' ? 'High-Yield' : doc.style}
                                    </Badge>
                                    <span className="text-xs text-muted-foreground">
                                      {doc._count?.mcqs || doc._count?.flashcards || 0} items
                                    </span>
                                  </div>
                                </div>
                              </div>
                              <div className="flex items-center gap-2">
                                <span className="text-xs text-muted-foreground">
                                  {new Date(doc.createdAt).toLocaleDateString()}
                                </span>
                                <Button
                                  variant="ghost"
                                  size="icon"
                                  onClick={(e) => {
                                    e.stopPropagation();
                                    handleDeleteDocument(doc.id);
                                  }}
                                  className="text-red-500 hover:text-red-700 hover:bg-red-50 dark:hover:bg-red-950"
                                >
                                  <Trash2 className="w-4 h-4" />
                                </Button>
                              </div>
                            </div>
                          </CardContent>
                        </Card>
                      ))}
                    </div>
                  </ScrollArea>
                )}
              </motion.div>
            )}

            {/* Bookmarks Tab */}
            {activeTab === 'bookmarks' && (
              <motion.div
                key="bookmarks"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                className="max-w-4xl mx-auto"
              >
                <div className="flex items-center justify-between mb-6">
                  <div>
                    <h2 className="text-2xl font-bold">Your Bookmarks</h2>
                    <p className="text-muted-foreground">{bookmarks.length} items saved</p>
                  </div>
                  <div className="flex items-center gap-2">
                    {bookmarks.length > 0 && (
                      <Button 
                        onClick={() => startBookmarkPractice('all')} 
                        className="gap-2"
                        disabled={bookmarks.length === 0}
                      >
                        <Play className="w-4 h-4" />
                        Practice All
                      </Button>
                    )}
                    <Button variant="outline" onClick={fetchBookmarks} className="gap-2">
                      <RefreshCw className="w-4 h-4" />
                      Refresh
                    </Button>
                  </div>
                </div>

                <Tabs value={bookmarkFilter} onValueChange={(v) => setBookmarkFilter(v as 'all' | 'mcq' | 'flashcard')} className="w-full mb-6">
                  <TabsList>
                    <TabsTrigger value="all">
                      All ({bookmarks.length})
                    </TabsTrigger>
                    <TabsTrigger value="mcq">
                      MCQs ({mcqBookmarkCount})
                    </TabsTrigger>
                    <TabsTrigger value="flashcard">
                      Flashcards ({flashcardBookmarkCount})
                    </TabsTrigger>
                  </TabsList>
                </Tabs>

                {bookmarks.length === 0 ? (
                  <Card className="border-dashed">
                    <CardContent className="flex flex-col items-center justify-center py-12">
                      <Bookmark className="w-12 h-12 text-muted-foreground mb-4" />
                      <h3 className="text-lg font-medium mb-2">No bookmarks yet</h3>
                      <p className="text-muted-foreground text-center mb-4">
                        Bookmark MCQs and flashcards to save them for later review.
                      </p>
                      <Button onClick={() => setActiveTab('home')} className="gap-2">
                        <Plus className="w-4 h-4" />
                        Generate Content
                      </Button>
                    </CardContent>
                  </Card>
                ) : filteredBookmarks.length === 0 ? (
                  <Card className="border-dashed">
                    <CardContent className="flex flex-col items-center justify-center py-12">
                      {bookmarkFilter === 'mcq' ? (
                        <>
                          <GraduationCap className="w-12 h-12 text-muted-foreground mb-4" />
                          <h3 className="text-lg font-medium mb-2">No MCQ bookmarks</h3>
                          <p className="text-muted-foreground text-center mb-4">
                            Bookmark MCQs while practicing to save them here.
                          </p>
                        </>
                      ) : (
                        <>
                          <Layers className="w-12 h-12 text-muted-foreground mb-4" />
                          <h3 className="text-lg font-medium mb-2">No Flashcard bookmarks</h3>
                          <p className="text-muted-foreground text-center mb-4">
                            Bookmark flashcards while practicing to save them here.
                          </p>
                        </>
                      )}
                      <Button onClick={() => setBookmarkFilter('all')} variant="outline">
                        View All Bookmarks
                      </Button>
                    </CardContent>
                  </Card>
                ) : (
                  <div className="space-y-6">
                    {/* MCQ Section */}
                    {(bookmarkFilter === 'all' || bookmarkFilter === 'mcq') && mcqBookmarkCount > 0 && (
                      <div className="space-y-4">
                        {bookmarkFilter === 'all' && (
                          <div className="flex items-center justify-between">
                            <div className="flex items-center gap-2">
                              <GraduationCap className="w-5 h-5 text-violet-500" />
                              <h3 className="font-semibold">MCQ Questions</h3>
                              <Badge variant="secondary">{mcqBookmarkCount}</Badge>
                            </div>
                            <Button 
                              variant="outline" 
                              size="sm" 
                              onClick={() => startBookmarkPractice('mcq')}
                              className="gap-2"
                            >
                              <Play className="w-4 h-4" />
                              Practice MCQs
                            </Button>
                          </div>
                        )}
                        {bookmarkFilter === 'mcq' && (
                          <div className="flex items-center justify-between">
                            <div className="flex items-center gap-2">
                              <GraduationCap className="w-5 h-5 text-violet-500" />
                              <span className="text-sm text-muted-foreground">{mcqBookmarkCount} MCQs</span>
                            </div>
                            <Button 
                              size="sm" 
                              onClick={() => startBookmarkPractice('mcq')}
                              className="gap-2"
                            >
                              <Play className="w-4 h-4" />
                              Practice MCQs
                            </Button>
                          </div>
                        )}
                        <ScrollArea className={bookmarkFilter === 'all' ? "h-[300px]" : "h-[calc(100vh-400px)]"}>
                          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pr-4">
                            {(bookmarkFilter === 'all' ? bookmarks.filter(b => b.itemType === 'mcq') : filteredBookmarks).map((bookmark) => (
                              <Card key={bookmark.id} className="overflow-hidden">
                                <CardHeader className="pb-3">
                                  <div className="flex items-start justify-between">
                                    <div className="flex items-center gap-2">
                                      <Badge className={difficultyColors[bookmark.item.difficulty as Difficulty] || difficultyColors.medium}>
                                        {bookmark.item.difficulty}
                                      </Badge>
                                      <Badge variant="secondary">
                                        {bookmark.item.style === 'high_yield' ? 'High-Yield' : bookmark.item.style}
                                      </Badge>
                                    </div>
                                    <Button
                                      variant="ghost"
                                      size="icon"
                                      onClick={() => removeBookmark(bookmark.id, bookmark.itemType, bookmark.itemId)}
                                      className="text-red-500 hover:text-red-700 hover:bg-red-50 dark:hover:bg-red-950"
                                    >
                                      <Trash2 className="w-4 h-4" />
                                    </Button>
                                  </div>
                                </CardHeader>
                                <CardContent>
                                  <div className="space-y-3">
                                    <p className="font-medium">{bookmark.item.question}</p>
                                    <div className="grid grid-cols-2 gap-2 text-sm">
                                      {(['A', 'B', 'C', 'D'] as const).map((letter) => (
                                        <div
                                          key={letter}
                                          className={cn(
                                            'p-2 rounded-lg border transition-colors',
                                            letter === bookmark.item.correctAnswer
                                              ? 'border-green-500 bg-green-50 dark:bg-green-950'
                                              : 'hover:bg-slate-50 dark:hover:bg-slate-800'
                                          )}
                                        >
                                          <span className="font-medium">{letter}.</span> {bookmark.item.options?.[letter]}
                                          {letter === bookmark.item.correctAnswer && (
                                            <Check className="inline-block ml-2 w-4 h-4 text-green-500" />
                                          )}
                                        </div>
                                      ))}
                                    </div>
                                    {bookmark.item.explanation && (
                                      <div className="p-3 bg-slate-50 dark:bg-slate-800 rounded-lg">
                                        <p className="text-sm font-medium mb-1">Explanation:</p>
                                        <p className="text-sm text-muted-foreground">{bookmark.item.explanation}</p>
                                      </div>
                                    )}
                                  </div>
                                </CardContent>
                              </Card>
                            ))}
                          </div>
                        </ScrollArea>
                      </div>
                    )}

                    {/* Flashcard Section */}
                    {(bookmarkFilter === 'all' || bookmarkFilter === 'flashcard') && flashcardBookmarkCount > 0 && (
                      <div className="space-y-4">
                        {bookmarkFilter === 'all' && (
                          <div className="flex items-center justify-between">
                            <div className="flex items-center gap-2">
                              <Layers className="w-5 h-5 text-emerald-500" />
                              <h3 className="font-semibold">Flashcards</h3>
                              <Badge variant="secondary">{flashcardBookmarkCount}</Badge>
                            </div>
                            <Button 
                              variant="outline" 
                              size="sm" 
                              onClick={() => startBookmarkPractice('flashcard')}
                              className="gap-2"
                            >
                              <Play className="w-4 h-4" />
                              Practice Flashcards
                            </Button>
                          </div>
                        )}
                        {bookmarkFilter === 'flashcard' && (
                          <div className="flex items-center justify-between">
                            <div className="flex items-center gap-2">
                              <Layers className="w-5 h-5 text-emerald-500" />
                              <span className="text-sm text-muted-foreground">{flashcardBookmarkCount} Flashcards</span>
                            </div>
                            <Button 
                              size="sm" 
                              onClick={() => startBookmarkPractice('flashcard')}
                              className="gap-2"
                            >
                              <Play className="w-4 h-4" />
                              Practice Flashcards
                            </Button>
                          </div>
                        )}
                        <ScrollArea className={bookmarkFilter === 'all' ? "h-[300px]" : "h-[calc(100vh-400px)]"}>
                          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pr-4">
                            {(bookmarkFilter === 'all' ? bookmarks.filter(b => b.itemType === 'flashcard') : filteredBookmarks).map((bookmark) => (
                              <Card key={bookmark.id} className="overflow-hidden">
                                <CardHeader className="pb-3">
                                  <div className="flex items-start justify-between">
                                    <div className="flex items-center gap-2">
                                      <Badge className={difficultyColors[bookmark.item.difficulty as Difficulty] || difficultyColors.medium}>
                                        {bookmark.item.difficulty}
                                      </Badge>
                                      <Badge variant="secondary">
                                        {bookmark.item.style === 'high_yield' ? 'High-Yield' : bookmark.item.style}
                                      </Badge>
                                    </div>
                                    <Button
                                      variant="ghost"
                                      size="icon"
                                      onClick={() => removeBookmark(bookmark.id, bookmark.itemType, bookmark.itemId)}
                                      className="text-red-500 hover:text-red-700 hover:bg-red-50 dark:hover:bg-red-950"
                                    >
                                      <Trash2 className="w-4 h-4" />
                                    </Button>
                                  </div>
                                </CardHeader>
                                <CardContent>
                                  <div className="space-y-3">
                                    <div className="p-3 bg-violet-50 dark:bg-violet-950 rounded-lg">
                                      <p className="text-xs font-medium text-violet-600 dark:text-violet-400 mb-1">Front</p>
                                      <p className="font-medium">{bookmark.item.front}</p>
                                    </div>
                                    <div className="p-3 bg-emerald-50 dark:bg-emerald-950 rounded-lg">
                                      <p className="text-xs font-medium text-emerald-600 dark:text-emerald-400 mb-1">Back</p>
                                      <div className="prose prose-sm dark:prose-invert max-w-none flashcard-markdown">
                                        <ReactMarkdown>{bookmark.item.back || ''}</ReactMarkdown>
                                      </div>
                                    </div>
                                  </div>
                                </CardContent>
                              </Card>
                            ))}
                          </div>
                        </ScrollArea>
                      </div>
                    )}
                  </div>
                )}
              </motion.div>
            )}

            {/* Practice Tab */}
            {activeTab === 'practice' && (
              <motion.div
                key="practice"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                className="max-w-3xl mx-auto"
              >
                {/* Practice Source Indicator */}
                {practiceSource === 'bookmarks' && (
                  <div className="mb-4 p-3 bg-violet-50 dark:bg-violet-950 rounded-lg flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <Bookmark className="w-4 h-4 text-violet-600" />
                      <span className="text-sm font-medium text-violet-900 dark:text-violet-100">
                        Practicing from Bookmarks
                      </span>
                    </div>
                    {/* Mode Switcher - only show if both types available */}
                    {mcqs.length > 0 && flashcards.length > 0 && (
                      <ToggleGroup
                        type="single"
                        value={mode}
                        onValueChange={(value) => {
                          if (value) {
                            setMode(value as Mode);
                            setPracticeIndex(0);
                            setAnswers({});
                            setResults({});
                            setShowExplanation(false);
                            setShowFlashcardBack(false);
                          }
                        }}
                        className="gap-1"
                      >
                        <ToggleGroupItem
                          value="mcq"
                          size="sm"
                          className="gap-1 data-[state=on]:bg-violet-100 data-[state=on]:text-violet-900"
                        >
                          <GraduationCap className="w-4 h-4" />
                          MCQ ({mcqs.length})
                        </ToggleGroupItem>
                        <ToggleGroupItem
                          value="flashcard"
                          size="sm"
                          className="gap-1 data-[state=on]:bg-violet-100 data-[state=on]:text-violet-900"
                        >
                          <Layers className="w-4 h-4" />
                          Cards ({flashcards.length})
                        </ToggleGroupItem>
                      </ToggleGroup>
                    )}
                  </div>
                )}

                {/* Progress Bar */}
                <div className="mb-6">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm font-medium">
                      {mode === 'mcq' ? 'Question' : 'Card'} {practiceIndex + 1} of {mode === 'mcq' ? mcqs.length : flashcards.length}
                    </span>
                    {mode === 'mcq' && Object.keys(results).length > 0 && (
                      <span className="text-sm text-muted-foreground">
                        Score: {calculateScore().correct}/{Object.keys(results).length}
                      </span>
                    )}
                  </div>
                  <Progress
                    value={((practiceIndex + 1) / (mode === 'mcq' ? mcqs.length : flashcards.length)) * 100}
                  />
                </div>

                {/* MCQ Practice */}
                {mode === 'mcq' && mcqs[practiceIndex] && (
                  <Card className="overflow-hidden">
                    <CardHeader>
                      <div className="flex items-start justify-between mb-2">
                        <div className="flex items-center gap-2">
                          <Badge className={difficultyColors[mcqs[practiceIndex].difficulty]}>
                            {mcqs[practiceIndex].difficulty}
                          </Badge>
                          <Badge variant="secondary">
                            {mcqs[practiceIndex].style === 'high_yield' ? 'High-Yield' : mcqs[practiceIndex].style}
                          </Badge>
                        </div>
                        <TooltipProvider>
                          <Tooltip>
                            <TooltipTrigger asChild>
                              <Button
                                variant="ghost"
                                size="icon"
                                onClick={() => toggleBookmark('mcq', mcqs[practiceIndex].id)}
                                className={cn(
                                  isBookmarked('mcq', mcqs[practiceIndex].id)
                                    ? 'text-violet-500 hover:text-violet-700 hover:bg-violet-50 dark:hover:bg-violet-950'
                                    : 'hover:bg-slate-50 dark:hover:bg-slate-800'
                                )}
                              >
                                {isBookmarked('mcq', mcqs[practiceIndex].id) ? (
                                  <BookmarkCheck className="w-5 h-5" />
                                ) : (
                                  <Bookmark className="w-5 h-5" />
                                )}
                              </Button>
                            </TooltipTrigger>
                            <TooltipContent>
                              {isBookmarked('mcq', mcqs[practiceIndex].id) ? 'Remove bookmark' : 'Add bookmark'}
                            </TooltipContent>
                          </Tooltip>
                        </TooltipProvider>
                      </div>
                      <CardTitle className="text-xl">{mcqs[practiceIndex].question}</CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-3">
                      {(['A', 'B', 'C', 'D'] as const).map((letter) => {
                        const isAnswered = answers[mcqs[practiceIndex].id] !== undefined;
                        const isSelected = answers[mcqs[practiceIndex].id] === letter;
                        const isCorrect = mcqs[practiceIndex].correctAnswer === letter;
                        const showResult = results[mcqs[practiceIndex].id] !== undefined;

                        return (
                          <button
                            key={letter}
                            disabled={showResult}
                            onClick={() => handleAnswer(mcqs[practiceIndex].id, letter)}
                            className={cn(
                              'w-full p-4 rounded-lg border-2 text-left transition-all',
                              !showResult && 'hover:border-violet-300 hover:bg-violet-50 dark:hover:bg-violet-950',
                              !showResult && isSelected && 'border-violet-500 bg-violet-50 dark:bg-violet-950',
                              showResult && isCorrect && 'border-green-500 bg-green-50 dark:bg-green-950',
                              showResult && isSelected && !isCorrect && 'border-red-500 bg-red-50 dark:bg-red-950',
                            )}
                          >
                            <span className="font-medium">{letter}.</span> {mcqs[practiceIndex].options[letter]}
                            {showResult && isCorrect && (
                              <Check className="inline-block ml-2 w-5 h-5 text-green-500" />
                            )}
                            {showResult && isSelected && !isCorrect && (
                              <X className="inline-block ml-2 w-5 h-5 text-red-500" />
                            )}
                          </button>
                        );
                      })}

                      {/* Hint Button */}
                      {!results[mcqs[practiceIndex].id] && mcqs[practiceIndex].hint && (
                        <Button
                          variant="ghost"
                          className="w-full gap-2 text-amber-600"
                          onClick={() => {
                            toast({
                              title: '💡 Hint',
                              description: mcqs[practiceIndex].hint,
                            });
                          }}
                        >
                          <Lightbulb className="w-4 h-4" />
                          Show Hint
                        </Button>
                      )}

                      {/* Explanation */}
                      {showExplanation && (
                        <motion.div
                          initial={{ opacity: 0, height: 0 }}
                          animate={{ opacity: 1, height: 'auto' }}
                          className="p-4 bg-slate-50 dark:bg-slate-800 rounded-lg"
                        >
                          <p className="font-medium mb-2">Explanation:</p>
                          <p className="text-muted-foreground">{mcqs[practiceIndex].explanation}</p>
                        </motion.div>
                      )}
                    </CardContent>
                    <CardFooter className="flex justify-between">
                      <Button
                        variant="outline"
                        onClick={prevQuestion}
                        disabled={practiceIndex === 0}
                        className="gap-2"
                      >
                        <ChevronLeft className="w-4 h-4" />
                        Previous
                      </Button>
                      {results[mcqs[practiceIndex].id] === undefined ? (
                        <Button
                          onClick={() => checkAnswer(mcqs[practiceIndex].id, mcqs[practiceIndex].correctAnswer)}
                          disabled={!answers[mcqs[practiceIndex].id]}
                          className="gap-2"
                        >
                          Check Answer
                        </Button>
                      ) : practiceIndex < mcqs.length - 1 ? (
                        <Button onClick={nextQuestion} className="gap-2">
                          Next
                          <ChevronRight className="w-4 h-4" />
                        </Button>
                      ) : (
                        <Button
                          onClick={() => setActiveTab(practiceSource === 'bookmarks' ? 'bookmarks' : 'results')}
                          className="gap-2"
                        >
                          Finish Practice
                        </Button>
                      )}
                    </CardFooter>
                  </Card>
                )}

                {/* Flashcard Practice */}
                {mode === 'flashcard' && flashcards[practiceIndex] && (
                  <div className="space-y-4">
                    {/* Flashcard Header with Bookmark */}
                    <div className="flex items-center justify-end">
                      <TooltipProvider>
                        <Tooltip>
                          <TooltipTrigger asChild>
                            <Button
                              variant="ghost"
                              size="icon"
                              onClick={() => toggleBookmark('flashcard', flashcards[practiceIndex].id)}
                              className={cn(
                                isBookmarked('flashcard', flashcards[practiceIndex].id)
                                  ? 'text-violet-500 hover:text-violet-700 hover:bg-violet-50 dark:hover:bg-violet-950'
                                  : 'hover:bg-slate-50 dark:hover:bg-slate-800'
                              )}
                            >
                              {isBookmarked('flashcard', flashcards[practiceIndex].id) ? (
                                <BookmarkCheck className="w-5 h-5" />
                              ) : (
                                <Bookmark className="w-5 h-5" />
                              )}
                            </Button>
                          </TooltipTrigger>
                          <TooltipContent>
                            {isBookmarked('flashcard', flashcards[practiceIndex].id) ? 'Remove bookmark' : 'Add bookmark'}
                          </TooltipContent>
                        </Tooltip>
                      </TooltipProvider>
                    </div>
                    <Card
                      className="min-h-[300px] cursor-pointer perspective-1000"
                      onClick={() => setShowFlashcardBack(!showFlashcardBack)}
                    >
                      <CardContent className="flex flex-col items-center justify-center p-8 min-h-[300px]">
                        <AnimatePresence mode="wait">
                          {!showFlashcardBack ? (
                            <motion.div
                              key="front"
                              initial={{ rotateY: 0 }}
                              exit={{ rotateY: 180, opacity: 0 }}
                              className="text-center"
                            >
                              <Badge className={cn(difficultyColors[flashcards[practiceIndex].difficulty], "mb-4")}>
                                {flashcards[practiceIndex].difficulty}
                              </Badge>
                              <p className="text-xl font-medium">{flashcards[practiceIndex].front}</p>
                              <p className="text-sm text-muted-foreground mt-4">
                                Click to reveal answer
                              </p>
                            </motion.div>
                          ) : (
                            <motion.div
                              key="back"
                              initial={{ rotateY: -180, opacity: 0 }}
                              animate={{ rotateY: 0, opacity: 1 }}
                              className="text-left w-full max-w-lg"
                            >
                              <div className="prose prose-base dark:prose-invert max-w-none flashcard-markdown">
                                <ReactMarkdown>{flashcards[practiceIndex].back}</ReactMarkdown>
                              </div>
                              {flashcards[practiceIndex].mnemonic && (
                                <div className="mt-4 p-3 bg-amber-50 dark:bg-amber-950 rounded-lg">
                                  <p className="text-sm font-medium text-amber-600 flex items-center gap-1">
                                    <Lightbulb className="w-4 h-4" />
                                    Mnemonic
                                  </p>
                                  <div className="prose prose-sm dark:prose-invert max-w-none mt-1 flashcard-markdown">
                                    <ReactMarkdown>{flashcards[practiceIndex].mnemonic}</ReactMarkdown>
                                  </div>
                                </div>
                              )}
                              {flashcards[practiceIndex].clinicalCorrelation && (
                                <div className="mt-2 p-3 bg-blue-50 dark:bg-blue-950 rounded-lg">
                                  <p className="text-sm font-medium text-blue-600 flex items-center gap-1">
                                    <Stethoscope className="w-4 h-4" />
                                    Clinical Correlation
                                  </p>
                                  <div className="prose prose-sm dark:prose-invert max-w-none mt-1 flashcard-markdown">
                                    <ReactMarkdown>{flashcards[practiceIndex].clinicalCorrelation}</ReactMarkdown>
                                  </div>
                                </div>
                              )}
                            </motion.div>
                          )}
                        </AnimatePresence>
                      </CardContent>
                    </Card>

                    <div className="flex justify-between">
                      <Button
                        variant="outline"
                        onClick={prevQuestion}
                        disabled={practiceIndex === 0}
                        className="gap-2"
                      >
                        <ChevronLeft className="w-4 h-4" />
                        Previous
                      </Button>
                      <Button
                        variant="outline"
                        onClick={() => setShowFlashcardBack(!showFlashcardBack)}
                        className="gap-2"
                      >
                        {showFlashcardBack ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                        {showFlashcardBack ? 'Hide' : 'Reveal'}
                      </Button>
                      {practiceIndex < flashcards.length - 1 ? (
                        <Button onClick={nextQuestion} className="gap-2">
                          Next
                          <ChevronRight className="w-4 h-4" />
                        </Button>
                      ) : (
                        <Button onClick={() => setActiveTab(practiceSource === 'bookmarks' ? 'bookmarks' : 'results')} className="gap-2">
                          Finish
                        </Button>
                      )}
                    </div>
                  </div>
                )}

                {/* Back Button */}
                <Button
                  variant="ghost"
                  onClick={() => setActiveTab(practiceSource === 'bookmarks' ? 'bookmarks' : 'results')}
                  className="mt-6 w-full"
                >
                  <RotateCcw className="w-4 h-4 mr-2" />
                  {practiceSource === 'bookmarks' ? 'Back to Bookmarks' : 'Back to Results'}
                </Button>
              </motion.div>
            )}
          </AnimatePresence>
        </main>

        {/* Footer */}
        <footer className="border-t bg-white/80 dark:bg-slate-900/80 backdrop-blur-sm py-4 mt-auto">
          <div className="container mx-auto px-4 flex items-center justify-between text-sm text-muted-foreground">
            <p>StudyAI - AI-Powered Learning Platform</p>
            <p>Built with Next.js & DeepSeek-V3 AI</p>
          </div>
        </footer>
      </div>
    </TooltipProvider>
  );
}
