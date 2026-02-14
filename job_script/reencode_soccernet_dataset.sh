#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ---------------------------------------------------------------------------
# Env overrides
# ---------------------------------------------------------------------------
# FFMPEG_BIN: ffmpeg executable (default: ffmpeg)
# SRC_ROOT: source SoccerNet root (default: ${REPO_ROOT}/SoccerNet)
# DST_ROOT: destination root for re-encoded videos (default: ${REPO_ROOT}/SoccerNet_reencoded)
#
# MODE:
#   all      -> re-encode every video found
#   bad_only -> first scan decode errors, then re-encode only problematic files
#   (default: all)
#
# OVERWRITE: true|false (default: false)
# CRF: x264 CRF quality (default: 18)
# PRESET: x264 preset (default: medium)
# GOP: keyframe interval (default: 50)
# AUDIO_BITRATE: AAC bitrate (default: 192k)
# LOG_DIR: where lists/logs are written (default: ${REPO_ROOT}/output/reencode_logs)

FFMPEG_BIN="${FFMPEG_BIN:-ffmpeg}"
SRC_ROOT="${SRC_ROOT:-${REPO_ROOT}/SoccerNet}"
DST_ROOT="${DST_ROOT:-${REPO_ROOT}/SoccerNet_reencoded}"
MODE="${MODE:-all}"
OVERWRITE="${OVERWRITE:-false}"
CRF="${CRF:-18}"
PRESET="${PRESET:-medium}"
GOP="${GOP:-50}"
AUDIO_BITRATE="${AUDIO_BITRATE:-192k}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/output/reencode_logs}"

mkdir -p "${LOG_DIR}"

ALL_LIST="${LOG_DIR}/all_videos.nul"
TODO_LIST="${LOG_DIR}/todo_videos.nul"
FAILED_LIST="${LOG_DIR}/failed_reencode.txt"
SCAN_ERRORS="${LOG_DIR}/scan_errors.txt"
MANIFEST="${LOG_DIR}/reencode_manifest.csv"

: > "${FAILED_LIST}"
: > "${SCAN_ERRORS}"
printf "input_path,output_path,status\n" > "${MANIFEST}"

if ! command -v "${FFMPEG_BIN}" >/dev/null 2>&1; then
  echo "[ERROR] ffmpeg not found: ${FFMPEG_BIN}" >&2
  exit 1
fi

if [[ ! -d "${SRC_ROOT}" ]]; then
  echo "[ERROR] Source root not found: ${SRC_ROOT}" >&2
  exit 1
fi

case "${MODE}" in
  all|bad_only) ;;
  *)
    echo "[ERROR] MODE must be one of: all, bad_only" >&2
    exit 1
    ;;
esac

echo "=============================================="
echo "SoccerNet Re-encode Job"
echo "=============================================="
echo "Source root:      ${SRC_ROOT}"
echo "Destination root: ${DST_ROOT}"
echo "Mode:             ${MODE}"
echo "Overwrite:        ${OVERWRITE}"
echo "CRF / Preset:     ${CRF} / ${PRESET}"
echo "GOP:              ${GOP}"
echo "Log dir:          ${LOG_DIR}"
echo "=============================================="

find "${SRC_ROOT}" -type f \( \
  -iname "*.mkv" -o \
  -iname "*.mp4" -o \
  -iname "*.avi" -o \
  -iname "*.mov" \
  \) -print0 > "${ALL_LIST}"

TOTAL_FOUND="$(tr -cd '\0' < "${ALL_LIST}" | wc -c | tr -d ' ')"
if [[ "${TOTAL_FOUND}" == "0" ]]; then
  echo "[ERROR] No video files found under ${SRC_ROOT}" >&2
  exit 1
fi
echo "[INFO] Found ${TOTAL_FOUND} video file(s)."

if [[ "${MODE}" == "all" ]]; then
  cp "${ALL_LIST}" "${TODO_LIST}"
else
  : > "${TODO_LIST}"
  echo "[INFO] Scanning videos for decode errors (ffmpeg -v error)..."
  while IFS= read -r -d '' in_path; do
    tmp_err="$(mktemp)"
    if ! "${FFMPEG_BIN}" -hide_banner -nostdin -v error -i "${in_path}" -f null - \
      >/dev/null 2>"${tmp_err}"; then
      :
    fi

    if [[ -s "${tmp_err}" ]]; then
      printf '%s\0' "${in_path}" >> "${TODO_LIST}"
      {
        echo "----- ${in_path}"
        cat "${tmp_err}"
      } >> "${SCAN_ERRORS}"
    fi
    rm -f "${tmp_err}"
  done < "${ALL_LIST}"
fi

TOTAL_TODO="$(tr -cd '\0' < "${TODO_LIST}" | wc -c | tr -d ' ')"
echo "[INFO] Videos selected for re-encode: ${TOTAL_TODO}"

if [[ "${TOTAL_TODO}" == "0" ]]; then
  echo "[INFO] Nothing to re-encode. Done."
  exit 0
fi

success_count=0
skip_count=0
fail_count=0
index=0

while IFS= read -r -d '' in_path; do
  index=$((index + 1))
  rel_path="${in_path#${SRC_ROOT}/}"
  out_path="${DST_ROOT}/${rel_path%.*}.mp4"

  mkdir -p "$(dirname "${out_path}")"

  if [[ -f "${out_path}" && "${OVERWRITE}" != "true" ]]; then
    skip_count=$((skip_count + 1))
    printf '"%s","%s","skipped_exists"\n' "${in_path}" "${out_path}" >> "${MANIFEST}"
    echo "[${index}/${TOTAL_TODO}] [SKIP] ${out_path} exists"
    continue
  fi

  echo "[${index}/${TOTAL_TODO}] [ENCODE] ${in_path}"
  if [[ "${OVERWRITE}" == "true" ]]; then
    overwrite_flag="-y"
  else
    overwrite_flag="-n"
  fi

  if "${FFMPEG_BIN}" -hide_banner -nostdin -loglevel warning "${overwrite_flag}" \
    -err_detect ignore_err \
    -i "${in_path}" \
    -map 0:v:0 -map 0:a? \
    -c:v libx264 \
    -preset "${PRESET}" \
    -crf "${CRF}" \
    -pix_fmt yuv420p \
    -g "${GOP}" \
    -keyint_min "${GOP}" \
    -sc_threshold 0 \
    -c:a aac \
    -b:a "${AUDIO_BITRATE}" \
    -movflags +faststart \
    "${out_path}"; then
    success_count=$((success_count + 1))
    printf '"%s","%s","ok"\n' "${in_path}" "${out_path}" >> "${MANIFEST}"
  else
    fail_count=$((fail_count + 1))
    echo "${in_path}" >> "${FAILED_LIST}"
    printf '"%s","%s","failed"\n' "${in_path}" "${out_path}" >> "${MANIFEST}"
    echo "[WARN] Failed: ${in_path}"
  fi
done < "${TODO_LIST}"

echo
echo "=============================================="
echo "Re-encode Summary"
echo "=============================================="
echo "Found videos:        ${TOTAL_FOUND}"
echo "Selected videos:     ${TOTAL_TODO}"
echo "Re-encoded success:  ${success_count}"
echo "Skipped existing:    ${skip_count}"
echo "Failed:              ${fail_count}"
echo "Manifest:            ${MANIFEST}"
echo "Failed list:         ${FAILED_LIST}"
if [[ "${MODE}" == "bad_only" ]]; then
  echo "Scan error log:      ${SCAN_ERRORS}"
fi
echo "=============================================="

