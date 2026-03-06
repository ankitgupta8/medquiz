import { config } from 'dotenv';
import { resolve } from 'path';

// Load .env with override before Prisma loads
config({ path: resolve(process.cwd(), '.env'), override: true });

// Fallback database URLs if env vars are not set
const DATABASE_URL = process.env.DATABASE_URL?.startsWith('postgresql') 
  ? process.env.DATABASE_URL 
  : "postgresql://postgres.fddwplujylgfosenajnu:Jp9848048293Rg@aws-1-ap-southeast-1.pooler.supabase.com:6543/postgres?pgbouncer=true";

// Set it explicitly
process.env.DATABASE_URL = DATABASE_URL;

import { PrismaClient } from '@prisma/client';

const globalForPrisma = globalThis as unknown as { prisma: PrismaClient | undefined };

export const db =
  globalForPrisma.prisma ??
  new PrismaClient({
    log: process.env.NODE_ENV === 'development' ? ['error', 'warn'] : ['error'],
  });

if (process.env.NODE_ENV !== 'production') globalForPrisma.prisma = db;
