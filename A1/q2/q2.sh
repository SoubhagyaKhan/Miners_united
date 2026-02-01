#!/bin/bash
# q2.sh - Final clean version
# Usage: bash q2.sh <gspan_exe> <fsg_exe> <gaston_exe> <dataset> <output_dir>

set -e

if [ "$#" -ne 5 ]; then
    echo "Usage: bash q2.sh <gspan> <fsg> <gaston> <dataset> <out>"
    exit 1
fi

GSPAN_EXEC=$1
FSG_EXEC=$2
GASTON_EXEC=$3
DATASET=$4
OUTPUT_DIR=$5

TIME_LIMIT="1h"
mkdir -p "$OUTPUT_DIR"

CONVERTED_DIR="${OUTPUT_DIR}/_converted"
mkdir -p "$CONVERTED_DIR"

echo "=============================================="
echo "Converting dataset formats..."
echo "=============================================="

python3 convert_data.py "$DATASET" "$CONVERTED_DIR"

FSG_DATA="${CONVERTED_DIR}/yeast_fsg.dat"
GSPAN_DATA="${CONVERTED_DIR}/yeast_gspan.dat"
GASTON_DATA="${CONVERTED_DIR}/yeast_gaston.dat"

TOTAL_GRAPHS=$(grep -c "^t #" "$GSPAN_DATA")
echo "Total graphs: $TOTAL_GRAPHS"

SUPPORTS=(5 10 25 50 95)

declare -A gspan_times
declare -A fsg_times
declare -A gaston_times

for SUP in "${SUPPORTS[@]}"; do
    echo ""
    echo "========== minSup = ${SUP}% =========="

    ABS_SUP=$(echo "($TOTAL_GRAPHS * $SUP) / 100" | bc)
    DEC_SUP=$(echo "scale=4; $SUP / 100" | bc)

    mkdir -p "$OUTPUT_DIR/gspan${SUP}"
    mkdir -p "$OUTPUT_DIR/fsg${SUP}"
    mkdir -p "$OUTPUT_DIR/gaston${SUP}"

    # ---------- FSG ----------
    echo "[FSG]"
    START=$(date +%s.%N)
    timeout "$TIME_LIMIT" "$FSG_EXEC" -s "$SUP" "$FSG_DATA" \
        > "$OUTPUT_DIR/fsg${SUP}/output.txt" 2>&1
    STATUS=$?
    END=$(date +%s.%N)

    if [ $STATUS -eq 124 ]; then
        fsg_times[$SUP]="TIMEOUT"
        echo "TIMEOUT"
    else
        fsg_times[$SUP]=$(echo "$END - $START" | bc)
        echo "${fsg_times[$SUP]} sec"
    fi

    # ---------- gSpan ----------
    echo "[gSpan]"
    START=$(date +%s.%N)
    timeout "$TIME_LIMIT" "$GSPAN_EXEC" -f "$GSPAN_DATA" -s "$DEC_SUP" \
        -o "$OUTPUT_DIR/gspan${SUP}/output.txt" 2>&1
    STATUS=$?
    END=$(date +%s.%N)

    if [ -f "${GSPAN_DATA}.fp" ]; then
        mv "${GSPAN_DATA}.fp" "$OUTPUT_DIR/gspan${SUP}/output.txt"
    fi

    if [ $STATUS -eq 124 ]; then
        gspan_times[$SUP]="TIMEOUT"
        echo "TIMEOUT"
    else
        gspan_times[$SUP]=$(echo "$END - $START" | bc)
        echo "${gspan_times[$SUP]} sec"
    fi

    # ---------- Gaston ----------
    echo "[Gaston]"
    START=$(date +%s.%N)
    timeout "$TIME_LIMIT" "$GASTON_EXEC" "$ABS_SUP" "$GASTON_DATA" \
        "$OUTPUT_DIR/gaston${SUP}/output.txt" 2>&1
    STATUS=$?
    END=$(date +%s.%N)

    if [ $STATUS -eq 124 ]; then
        gaston_times[$SUP]="TIMEOUT"
        echo "TIMEOUT"
    else
        gaston_times[$SUP]=$(echo "$END - $START" | bc)
        echo "${gaston_times[$SUP]} sec"
    fi
done

echo ""
echo "============== SUMMARY =============="
printf "%-10s %-12s %-12s %-12s\n" "Support" "FSG" "gSpan" "Gaston"

for SUP in 5 10 25 50 95; do
    printf "%-10s %-12s %-12s %-12s\n" \
        "$SUP" \
        "${fsg_times[$SUP]}" \
        "${gspan_times[$SUP]}" \
        "${gaston_times[$SUP]}"
done

cat > "$OUTPUT_DIR/timings.json" << EOF
{
  "fsg":   { "5": "${fsg_times[5]}",   "10": "${fsg_times[10]}",   "25": "${fsg_times[25]}",   "50": "${fsg_times[50]}",   "95": "${fsg_times[95]}" },
  "gspan": { "5": "${gspan_times[5]}", "10": "${gspan_times[10]}", "25": "${gspan_times[25]}", "50": "${gspan_times[50]}", "95": "${gspan_times[95]}" },
  "gaston":{ "5": "${gaston_times[5]}", "10": "${gaston_times[10]}", "25": "${gaston_times[25]}", "50": "${gaston_times[50]}", "95": "${gaston_times[95]}" }
}
EOF

python3 plot_q2.py "$OUTPUT_DIR/timings.json" "$OUTPUT_DIR/plot.png"
rm -rf "$CONVERTED_DIR"

echo ""
echo "DONE. Outputs in $OUTPUT_DIR"
