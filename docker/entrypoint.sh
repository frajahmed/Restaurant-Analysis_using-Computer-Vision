#!/bin/sh
# Container start script.
# On the first start the data volume is empty: copy the default configuration
# into it, so settings changed on the settings page are kept across updates.
set -e

if [ -n "$RCV_CONFIG" ] && [ ! -f "$RCV_CONFIG" ]; then
    mkdir -p "$(dirname "$RCV_CONFIG")"
    cp /app/config.json "$RCV_CONFIG"
    echo "Standard-Konfiguration angelegt: $RCV_CONFIG"
fi

exec "$@"
