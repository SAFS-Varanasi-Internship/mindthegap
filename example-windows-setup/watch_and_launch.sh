#!/usr/bin/env bash
# Poll the SkyPilot API health endpoint once a minute (a plain curl loop, no model
# tokens) and, the instant the backend responds instead of 503, launch the training
# run. Gives up after ~2 hours.
cd /home/truss/projects/skypilot-accelerator-mindthegap
PIXI="$HOME/.pixi/bin/pixi"
URL="https://cloudbank-skypilot.westus2.cloudapp.azure.com/api/health"
for i in $(seq 1 120); do
  code=$(curl -s -o /dev/null -w "%{http_code}" -m 15 "$URL")
  if [ "$code" -ge 200 ] 2>/dev/null && [ "$code" -lt 500 ] 2>/dev/null; then
    echo "SkyPilot API responding (HTTP $code) after $i checks; launching training..."
    "$PIXI" run remote-train -y
    exit $?
  fi
  echo "check $i: API still down (HTTP $code); sleeping 60s"
  sleep 60
done
echo "SkyPilot API still down after ~120 checks (~2h); giving up. Retry later or ping Scott."
exit 1
