# Use rocker/shiny-verse as base image
FROM rocker/shiny-verse:latest

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    && rm -rf /var/lib/apt/lists/*

# Install required R packages
RUN R -e "install.packages(c('shiny', 'shinydashboard', 'shinyjs', 'DT', 'plotly', 'ggplot2', 'httr', 'jsonlite', 'dplyr', 'lubridate'), repos='https://cran.rstudio.com/')"

# Create the app directory
RUN mkdir -p /srv/shiny-server/hotel-ranking

# Copy the Shiny app
COPY app.R /srv/shiny-server/hotel-ranking/

# Create startup script to ensure environment variables are available
RUN echo '#!/bin/bash\n\
echo "Starting Shiny Server with API_URL: $API_URL"\n\
export API_URL\n\
# Create .Renviron file for the shiny user\n\
echo "API_URL=$API_URL" > /home/shiny/.Renviron\n\
chown shiny:shiny /home/shiny/.Renviron\n\
exec /usr/bin/shiny-server' > /usr/local/bin/start-shiny.sh

# Make startup script executable
RUN chmod +x /usr/local/bin/start-shiny.sh

# Set permissions
RUN chown -R shiny:shiny /srv/shiny-server/hotel-ranking/

# Configure Shiny Server
RUN echo 'run_as shiny;\n\
server {\n\
  listen 3838;\n\
  location /hotel-ranking {\n\
    site_dir /srv/shiny-server/hotel-ranking;\n\
    log_dir /var/log/shiny-server;\n\
    directory_index on;\n\
  }\n\
}' > /etc/shiny-server/shiny-server.conf

# Expose port
EXPOSE 3838

# Start Shiny Server with environment variable support
CMD ["/usr/local/bin/start-shiny.sh"]

