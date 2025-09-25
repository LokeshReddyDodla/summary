/**
 * Cloudflare Worker for Report Summarization Bot
 * Proxies requests to Railway backend and adds CDN capabilities
 */

const RAILWAY_URL = 'https://your-railway-app-url.up.railway.app';

addEventListener('fetch', event => {
  event.respondWith(handleRequest(event.request));
});

async function handleRequest(request) {
  const url = new URL(request.url);
  
  // Handle CORS preflight requests
  if (request.method === 'OPTIONS') {
    return new Response(null, {
      status: 200,
      headers: {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
        'Access-Control-Allow-Headers': '*',
        'Access-Control-Max-Age': '86400',
      },
    });
  }

  // Proxy API requests to Railway
  if (url.pathname.startsWith('/api/') || 
      url.pathname === '/summarize-report' || 
      url.pathname.startsWith('/download-summary/') ||
      url.pathname === '/docs' ||
      url.pathname === '/openapi.json') {
    
    const targetUrl = RAILWAY_URL + url.pathname + url.search;
    
    const response = await fetch(targetUrl, {
      method: request.method,
      headers: {
        ...request.headers,
        'X-Forwarded-Host': url.host,
        'X-Forwarded-Proto': url.protocol.replace(':', ''),
      },
      body: request.method !== 'GET' && request.method !== 'HEAD' ? request.body : undefined,
    });

    // Add CORS headers to the response
    const newResponse = new Response(response.body, {
      status: response.status,
      statusText: response.statusText,
      headers: {
        ...response.headers,
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Credentials': 'true',
      },
    });

    return newResponse;
  }

  // Serve static files or return a simple HTML page
  if (url.pathname === '/' || url.pathname === '/index.html') {
    return new Response(`
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Report Summarization Bot</title>
    <style>
        body { 
            font-family: Arial, sans-serif; 
            margin: 2rem auto; 
            max-width: 720px; 
            line-height: 1.5; 
            color: #222; 
            text-align: center;
        }
        .container { 
            border: 1px solid #ccc; 
            padding: 2rem; 
            border-radius: 8px; 
            background-color: #fafafa; 
        }
        .status { color: #047857; }
    </style>
</head>
<body>
    <div class="container">
        <h1>📄 Report Summarization Bot</h1>
        <p class="status">✅ Service is running!</p>
        <p>Use the API endpoints:</p>
        <ul style="text-align: left; display: inline-block;">
            <li><code>POST /summarize-report</code> - Upload files and generate summary</li>
            <li><code>GET /download-summary/{token}</code> - Download generated summary</li>
            <li><code>GET /docs</code> - Interactive API documentation</li>
        </ul>
        <p><a href="/docs">View API Documentation</a></p>
    </div>
</body>
</html>`, {
      headers: {
        'Content-Type': 'text/html',
        'Cache-Control': 'public, max-age=300',
      },
    });
  }

  // Default response
  return new Response('Report Summarization Bot API', {
    headers: {
      'Content-Type': 'text/plain',
      'Cache-Control': 'public, max-age=300',
    },
  });
}
