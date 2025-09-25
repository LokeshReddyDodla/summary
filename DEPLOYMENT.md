# 🚀 Deployment Guide: Report Summarization Bot

This guide covers deploying the FastAPI application to Railway and Cloudflare for production use.

## Prerequisites

1. **Railway Account**: [railway.app](https://railway.app)
2. **Cloudflare Account**: [cloudflare.com](https://cloudflare.com)
3. **Custom Domain** (optional, but recommended)
4. **OpenAI API Key**: Get from [OpenAI Platform](https://platform.openai.com/api-keys)

## 🚂 Railway Deployment

### Step 1: Connect Repository
1. Go to [Railway Dashboard](https://railway.app/dashboard)
2. Click "New Project" → "Deploy from GitHub repo"
3. Connect your GitHub account and select the `summary` repository

### Step 2: Configure Environment Variables
In Railway project settings, add these environment variables:

```bash
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o-mini
CHUNK_TOKEN_LIMIT=2500
CHUNK_OVERLAP=200
```

### Step 3: Deploy
Railway will automatically detect the Python app and deploy it using the `railway.json` configuration.

### Step 4: Get Railway URL
After deployment, copy the Railway URL (e.g., `https://your-app-name.up.railway.app`)

## ☁️ Cloudflare Setup

### Option 1: Cloudflare Workers (Recommended)

1. **Install Wrangler CLI**:
   ```bash
   npm install -g wrangler
   ```

2. **Login to Cloudflare**:
   ```bash
   wrangler auth login
   ```

3. **Update Worker Configuration**:
   Edit `cloudflare/wrangler.toml`:
   ```toml
   # Replace with your Railway URL
   routes = [
     { pattern = "summary.yourdomain.com", zone_name = "yourdomain.com" }
   ]
   ```

4. **Deploy Worker**:
   ```bash
   cd cloudflare
   wrangler deploy
   ```

5. **Set Environment Variable**:
   ```bash
   wrangler secret put RAILWAY_URL
   # Enter: https://your-railway-app-url.up.railway.app
   ```

### Option 2: Cloudflare Pages (Alternative)

1. Go to [Cloudflare Pages](https://pages.cloudflare.com/)
2. Connect your GitHub repository
3. Set build command: `pip install -r requirements.txt && uvicorn main:app --host 0.0.0.0 --port 8000`
4. Configure environment variables in Pages settings

## 🌐 Domain Configuration

### Custom Domain Setup

1. **In Cloudflare**:
   - Add your domain to Cloudflare
   - Create a CNAME record pointing to your Railway/Cloudflare Workers URL

2. **SSL Certificate**:
   - Cloudflare provides free SSL certificates
   - Enable "Always Use HTTPS"

## 🔧 Environment Variables Reference

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `OPENAI_API_KEY` | OpenAI API key | Yes | - |
| `OPENAI_MODEL` | GPT model to use | No | `gpt-4o-mini` |
| `CHUNK_TOKEN_LIMIT` | Max tokens per chunk | No | `2500` |
| `CHUNK_OVERLAP` | Token overlap between chunks | No | `200` |
| `PORT` | Server port (Railway auto-sets) | No | `8000` |

## 📊 Monitoring & Scaling

### Railway Features
- **Auto-scaling**: Railway automatically scales based on traffic
- **Logs**: Access logs in Railway dashboard
- **Metrics**: Monitor CPU, memory, and request counts

### Cloudflare Benefits
- **CDN**: Global content delivery
- **DDoS Protection**: Automatic protection
- **Caching**: Improved performance
- **Analytics**: Request analytics and insights

## 🧪 Testing Deployment

### Test API Endpoints
```bash
# Test health check
curl https://your-domain.com/

# Test API docs
curl https://your-domain.com/docs

# Test summarization (replace with actual file)
curl -X POST https://your-domain.com/summarize-report \
  -F "files=@sample.txt"
```

### Test Web Interface
1. Visit: `https://your-domain.com/static/test-client.html`
2. Upload a document
3. Verify summarization works

## 🔒 Security Considerations

1. **API Key Protection**: Never expose OpenAI API key in client-side code
2. **Rate Limiting**: Consider implementing rate limiting for production
3. **File Upload Limits**: Railway has file size limits (check documentation)
4. **CORS**: Update CORS origins for production domains

## 🚨 Troubleshooting

### Common Issues

1. **Worker Deployment Fails**:
   ```bash
   wrangler tail  # Check logs
   ```

2. **Railway Build Fails**:
   - Check Railway logs in dashboard
   - Verify `requirements.txt` is correct
   - Ensure Python version compatibility

3. **API Key Issues**:
   - Verify environment variables are set correctly
   - Check OpenAI API key validity and credits

4. **CORS Errors**:
   - Update CORS origins in `main.py` for production domains
   - Clear browser cache

### Useful Commands

```bash
# Railway logs
railway logs

# Cloudflare Worker logs
wrangler tail

# Test Railway locally
railway run python main.py

# Deploy updates
git push origin main
```

## 📞 Support

- **Railway**: [railway.app/docs](https://docs.railway.app/)
- **Cloudflare**: [developers.cloudflare.com](https://developers.cloudflare.com/)
- **FastAPI**: [fastapi.tiangolo.com](https://fastapi.tiangolo.com/)

---

🎉 **Your Report Summarization Bot is now deployed and ready for production use!**
