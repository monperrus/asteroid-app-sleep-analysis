#!/bin/sh
# Start the persistent systemd recording service from an AsteroidOS launcher tile.

set -eu

unit=sleep-recorder.service
service=org.freedesktop.Notifications
path=/org/freedesktop/Notifications
export DBUS_SESSION_BUS_ADDRESS=${DBUS_SESSION_BUS_ADDRESS:-unix:path=/run/user/$(id -u)/bus}

if systemctl is-active --quiet "$unit"; then
    summary="Sleep recording is running"
    body="Collection is already active."
    result=0
elif printf '\n' | sudo -S systemctl start "$unit" && systemctl is-active --quiet "$unit"; then
    summary="Sleep recording is running"
    body="Movement and heart rate are being collected."
    result=0
else
    summary="Sleep recording did not start"
    body="Check the battery and try again."
    result=1
fi

# Asteroid's notification preview is short-lived. Close it explicitly in
# case the shell ignores the requested expiry. No display policy is changed.
if reply=$(busctl --user --timeout=2s call "$service" "$path" "$service" Notify \
        'susssasa{sv}i' 'Sleep Recorder' 0 ios-moon-outline "$summary" "$body" \
        0 5 x-nemo-preview-summary s "$summary" x-nemo-preview-body s "$body" \
        x-nemo-feedback s notif_silent urgency i 3 transient b true 4000); then
    case "$reply" in
        'u '*)
            sleep 4
            busctl --user --timeout=2s call "$service" "$path" "$service" \
                CloseNotification u "${reply#u }" >/dev/null || true
            ;;
    esac
fi
exit "$result"
