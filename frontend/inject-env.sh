#!/bin/sh
# Inject environment variables into app.js

# Use localhost for browser access (browser can't access Docker service names)
# Note: API_BASE is now hardcoded in app.js to avoid browser caching issues
# This script is kept for potential future use but doesn't modify API_BASE anymore

echo "✅ Frontend startup script executed (API_BASE is hardcoded in app.js)"

# Start nginx
exec nginx -g "daemon off;"

