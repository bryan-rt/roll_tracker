FROM debian:stable-slim

# Install tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg curl jq tzdata ca-certificates bash \
  && rm -rf /var/lib/apt/lists/*

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8

# Workdir & scripts
WORKDIR /app
COPY entrypoint.sh get_access_token.sh record_window.sh \
     list_cameras.sh register_cameras.sh diag_v*.sh /app/
RUN chmod +x /app/*.sh

# Data mount point
VOLUME ["/recordings"]

# Timezone comes from env TZ
ENTRYPOINT ["/app/entrypoint.sh"]
