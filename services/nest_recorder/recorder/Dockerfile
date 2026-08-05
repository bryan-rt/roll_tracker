# Pinned to trixie (Debian 13) — ships ffmpeg 7.x with -timeout, -fps_mode, -copyts.
# Do NOT use debian:stable-slim — it floats across releases and silently broke -stimeout
# when bookworm rolled to trixie. The startup assertion (check_ffmpeg_opts.sh) is the
# actual protection; this pin reduces churn.
FROM debian:trixie-slim

# Install tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg curl jq tzdata ca-certificates bash \
  && rm -rf /var/lib/apt/lists/*

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8

# Workdir & scripts
WORKDIR /app
# Copy ALL .sh scripts — wildcard prevents new scripts (e.g. check_ffmpeg_opts.sh)
# from being silently absent in the production image while present in dev via bind mount.
COPY *.sh /app/
RUN chmod +x /app/*.sh

# Data mount point
VOLUME ["/recordings"]

# Timezone comes from env TZ
ENTRYPOINT ["/app/entrypoint.sh"]
