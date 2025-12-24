FROM nginx:alpine

# Install sed for environment variable injection
RUN apk add --no-cache sed

# Copy frontend files
COPY . /usr/share/nginx/html

# Copy nginx configuration
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copy startup script
COPY inject-env.sh /docker-entrypoint.d/inject-env.sh
RUN chmod +x /docker-entrypoint.d/inject-env.sh

# Expose port
EXPOSE 80

# Use the startup script
CMD ["/docker-entrypoint.d/inject-env.sh"]

