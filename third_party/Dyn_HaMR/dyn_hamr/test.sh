#!/bin/bash

# Point BASE_PATH directly to subject1
BASE_PATH="/data/home/zy3023/code/hand/slam-hand/test/HOT3D"
VIDEO_JSON="/data/home/zy3023/code/hand/slam-hand/slahmr-eccv/confs/data/video.yaml"

# Directory for log files
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

# How many Python processes to run in parallel at most
MAX_JOBS=2

count=0  # Tracks how many jobs have been launched

for mp4file in "$BASE_PATH"/*.mp4; do
    [ -f "$mp4file" ] || continue

    # basename "a.mp4" => "a.mp4"
    mp4basename=$(basename "$mp4file")
    # strip the ".mp4" extension => "a"
    vidname="${mp4basename%.mp4}"

    logfile="${LOG_DIR}/${vidname}.log"

    # Skip this video if the log file exists and is non-empty
    if [ -s "$logfile" ]; then
        echo "Skipping $vidname, log file exists and is non-empty: $logfile"
        continue
    fi

    # Update 'seq' in the YAML to "a" instead of "a.mp4"
    sed -i "s|seq: *\"[^\"]*\"|seq: \"$vidname\"|" "$VIDEO_JSON"

    echo "Launching job for: $vidname => seq: \"$vidname\". Logs: $logfile"

    # Run Python in the background (&).
    # Output goes to "$logfile"
    python -u run_opt.py data=video run_opt=True run_vis=True data.seq=$vidname \
        > "$logfile" 2>&1 &

    ((count++))
    # If we've launched MAX_JOBS in parallel, wait for them to complete
    if (( count % MAX_JOBS == 0 )); then
        echo "Reached $count jobs. Waiting for them to finish..."
        wait
        echo "Continuing..."
    fi
done

# After the loop, wait for any remaining jobs
wait

echo "All done."
