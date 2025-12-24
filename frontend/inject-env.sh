#!/bin/sh
# Inject environment variables into app.js

# Use localhost for browser access (browser can't access Docker service names)
API_BASE_URL=${API_BASE_URL:-http://localhost:8000/api}

# Replace the API_BASE constant in app.js
sed -i "s|const API_BASE = 'http://localhost:8000/api';|const API_BASE = '${API_BASE_URL}';|g" /usr/share/nginx/html/app.js

echo "✅ Injected API_BASE_URL: ${API_BASE_URL}"

# Start nginx
exec nginx -g "daemon off;"

