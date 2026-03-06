#!/bin/bash
# Script to run prisma commands with correct environment variables

export NEXT_PUBLIC_SUPABASE_URL="https://fddwplujylgfosenajnu.supabase.co"
export NEXT_PUBLIC_SUPABASE_ANON_KEY="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImZkZHdwbHVqeWxnZm9zZW5ham51Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzI1ODMxNDIsImV4cCI6MjA4ODE1OTE0Mn0.A84IWE9K15_H_NLFDfaLyy67iXk342lnwYB-mdAT4HY"
export DATABASE_URL="postgresql://postgres.fddwplujylgfosenajnu:Jp9848048293Rg@aws-1-ap-southeast-1.pooler.supabase.com:6543/postgres?pgbouncer=true"
export DIRECT_URL="postgresql://postgres.fddwplujylgfosenajnu:Jp9848048293Rg@aws-1-ap-southeast-1.pooler.supabase.com:5432/postgres"
export MISTRAL_API_KEY="9tWzQBXD0jYxWkfaMGvLs0fXFMXAQ2d0"

bunx prisma db push "$@"
