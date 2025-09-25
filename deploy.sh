#!/bin/bash

# Report Summarization Bot - Deployment Script
# This script helps deploy the application to Railway and Cloudflare

set -e

echo "🚀 Report Summarization Bot - Deployment Script"
echo "=============================================="

# Check if Railway CLI is installed
if ! command -v railway &> /dev/null; then
    echo "❌ Railway CLI not found. Install it first:"
    echo "curl -fsSL https://railway.app/install.sh | sh"
    exit 1
fi

# Check if Wrangler CLI is installed
if ! command -v wrangler &> /dev/null; then
    echo "❌ Wrangler CLI not found. Install it first:"
    echo "npm install -g wrangler"
    exit 1
fi

echo "✅ CLI tools found"

# Deploy to Railway
echo ""
echo "🚂 Deploying to Railway..."
echo "=========================="

if [ ! -f ".env" ]; then
    echo "⚠️  .env file not found. Please create one with your OpenAI API key:"
    echo "cp .env.example .env"
    echo "Then edit .env with your actual API key"
    exit 1
fi

# Link to Railway project (you may need to create one first)
railway link || {
    echo "❌ Failed to link Railway project. Please:"
    echo "1. Go to https://railway.app/dashboard"
    echo "2. Create a new project"
    echo "3. Connect your GitHub repository"
    echo "4. Run this script again"
    exit 1
}

# Deploy
railway deploy

# Get Railway URL
RAILWAY_URL=$(railway domain)

echo "✅ Railway deployment complete!"
echo "🌐 Railway URL: $RAILWAY_URL"

# Cloudflare Workers deployment
echo ""
echo "☁️  Deploying Cloudflare Worker..."
echo "=================================="

cd cloudflare

# Update wrangler.toml with Railway URL
sed -i.bak "s|https://your-railway-app-url.up.railway.app|$RAILWAY_URL|g" wrangler.toml

# Deploy to Cloudflare
wrangler deploy

echo "✅ Cloudflare deployment complete!"
echo ""
echo "🎉 Deployment Summary:"
echo "======================"
echo "Railway Backend: $RAILWAY_URL"
echo "Cloudflare Worker: Check your Cloudflare dashboard for the URL"
echo ""
echo "📖 Next steps:"
echo "1. Set up your custom domain in Cloudflare"
echo "2. Configure DNS records"
echo "3. Test the API endpoints"
echo ""
echo "📚 See DEPLOYMENT.md for detailed instructions"
