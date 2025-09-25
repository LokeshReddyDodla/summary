# 🌐 Cloudflare Pages Setup Guide

This guide will help you deploy your Report Summarization Bot frontend to Cloudflare Pages with your Railway backend.

## 📋 Prerequisites

1. **Cloudflare Account**: [cloudflare.com](https://cloudflare.com)
2. **Railway Backend**: ✅ Already deployed at `summary-production-0277.up.railway.app`
3. **GitHub Repository**: Your `summary` repository

## 🚀 Cloudflare Pages Deployment

### Method 1: Direct Upload (Quickest)

1. **Go to Cloudflare Pages**:
   - Visit [pages.cloudflare.com](https://pages.cloudflare.com)
   - Click "Create a project" → "Upload assets"

2. **Upload Files**:
   - Drag and drop the `cloudflare-pages/` folder contents:
     - `index.html` (your frontend)
     - `_headers` (security headers)
     - `_redirects` (API proxying)

3. **Configure Project**:
   - Project name: `report-summarization-bot`
   - Production branch: `main`

4. **Deploy**:
   - Cloudflare will provide a URL like: `https://report-summarization-bot.pages.dev`

### Method 2: GitHub Integration (Recommended)

1. **Commit Frontend Files**:
   ```bash
   git add cloudflare-pages/ static/test-client.html
   git commit -m "Add Cloudflare Pages frontend setup"
   git push origin main
   ```

2. **Connect Repository**:
   - Go to [pages.cloudflare.com](https://pages.cloudflare.com)
   - Click "Create a project" → "Connect to Git"
   - Select your `summary` repository

3. **Build Configuration**:
   - Framework preset: `None`
   - Build command: `cp cloudflare-pages/* ./`
   - Build output directory: `/`
   - Root directory: `/`

4. **Environment Variables** (if needed):
   - `RAILWAY_BACKEND_URL`: `https://summary-production-0277.up.railway.app`

## 🔧 Configuration Details

### Frontend Features
- ✅ **Pre-configured** with your Railway backend URL
- ✅ **Security headers** for production
- ✅ **API proxying** through Cloudflare
- ✅ **File upload** interface
- ✅ **Download** functionality

### API Endpoints (Proxied)
- `POST /summarize-report` → Railway backend
- `GET /download-summary/{token}` → Railway backend
- `GET /docs` → Railway API documentation

### Security Headers
- X-Frame-Options: DENY
- X-Content-Type-Options: nosniff
- Referrer-Policy: strict-origin-when-cross-origin
- Permissions-Policy: camera=(), microphone=(), geolocation=()

## 🌍 Custom Domain (Optional)

1. **Add Domain to Cloudflare**:
   - Go to Cloudflare Dashboard → "Websites" → "Add a site"
   - Enter your domain name

2. **Configure Pages Custom Domain**:
   - In Pages project → "Custom domains"
   - Add your domain (e.g., `summary.yourdomain.com`)

3. **DNS Configuration**:
   - Cloudflare will automatically configure DNS
   - SSL certificate will be provisioned

## 🧪 Testing Your Deployment

### Test Frontend
1. Visit your Cloudflare Pages URL
2. Upload a test document (PDF, DOCX, or TXT)
3. Click "Summarize"
4. Download the generated summary

### Test API Integration
```bash
# Test via Cloudflare frontend
curl https://your-pages-url.pages.dev

# Direct backend test
curl https://summary-production-0277.up.railway.app/health
```

## 📊 Architecture Overview

```
User Browser
    ↓
Cloudflare Pages (Frontend)
    ↓ (API calls)
Railway Backend (FastAPI)
    ↓ (AI processing)
OpenAI API
```

## 🔒 Security Features

- **HTTPS Everywhere**: Both Cloudflare and Railway use SSL
- **CORS Protection**: Configured for your domains
- **File Upload Validation**: Client and server-side
- **API Rate Limiting**: Through Cloudflare
- **DDoS Protection**: Cloudflare's built-in protection

## 🚀 Benefits of This Setup

1. **Global CDN**: Fast loading worldwide
2. **Free Hosting**: Cloudflare Pages is free
3. **Auto-scaling**: Both platforms scale automatically
4. **SSL/TLS**: Automatic HTTPS certificates
5. **High Availability**: 99.9% uptime
6. **Easy Updates**: Git-based deployments

## 📞 Troubleshooting

### Common Issues

1. **CORS Errors**:
   - Check Railway backend CORS settings
   - Ensure frontend URL is in allowed origins

2. **API Not Working**:
   - Verify Railway backend is running
   - Check `_redirects` file configuration

3. **File Upload Fails**:
   - Check file size limits (Railway: 100MB)
   - Verify file type restrictions

### Useful Commands

```bash
# Test Railway backend
curl https://summary-production-0277.up.railway.app/health

# Check Cloudflare Pages deployment
curl https://your-pages-url.pages.dev

# Test file upload
curl -X POST https://your-pages-url.pages.dev/summarize-report \
  -F "files=@sample.txt"
```

---

🎉 **Your Report Summarization Bot is now deployed on a global, scalable infrastructure!**

**Frontend**: Cloudflare Pages (Global CDN)  
**Backend**: Railway (Auto-scaling API)  
**AI**: OpenAI GPT (Document processing)
