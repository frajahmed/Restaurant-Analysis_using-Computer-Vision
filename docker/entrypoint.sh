#!/bin/sh
# Container start script.
# On the first start the data volume is empty: copy the default configuration
# into it, so settings changed on the settings page are kept across updates.
set -e

DATA_DIR="$(dirname "${RCV_CONFIG:-/app/data/config.json}")"
mkdir -p "$DATA_DIR" 2>/dev/null || true

# A bind-mounted host folder may belong to a different user than the container
# user (uid 1000). Fail with a clear message instead of a cryptic traceback.
if [ ! -w "$DATA_DIR" ]; then
    echo "FEHLER: Keine Schreibrechte auf $DATA_DIR (Container-Benutzer uid $(id -u))."
    echo "Lösung: das Volume aus docker-compose.yml verwenden oder auf dem Host"
    echo "        'sudo chown -R 1000:1000 <ordner>' ausführen."
    exit 1
fi

if [ -n "$RCV_CONFIG" ] && [ ! -f "$RCV_CONFIG" ]; then
    cp /app/config.json "$RCV_CONFIG"
    echo "Standard-Konfiguration angelegt: $RCV_CONFIG"
fi

exec "$@"
