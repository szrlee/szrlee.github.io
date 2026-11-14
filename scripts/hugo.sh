#!/usr/bin/env bash
# Run the same Hugo version CI uses (WC_HUGO_VERSION in .github/workflows/publish.yaml).
# The theme (Hugo Blox v5) does not build with newer Hugo releases, so a system-wide Hugo from
# Homebrew may fail. On first use this downloads the official extended release into
# .cache/hugo/<version>/ (git-ignored), checks it against the release checksums, then runs it.
#
#   scripts/hugo.sh server          # local preview at http://localhost:1313
#   scripts/hugo.sh --gc --minify   # production build into public/
#   scripts/hugo.sh version
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
version="$(sed -n "s/^ *WC_HUGO_VERSION: *['\"]\{0,1\}\([0-9.]*\).*/\1/p" "$root/.github/workflows/publish.yaml" | head -1)"
[ -n "$version" ] || { echo "Could not read WC_HUGO_VERSION from .github/workflows/publish.yaml" >&2; exit 1; }

case "$(uname -s)-$(uname -m)" in
  Darwin-*)                  asset="hugo_extended_${version}_darwin-universal.tar.gz" ;;
  Linux-x86_64)              asset="hugo_extended_${version}_linux-amd64.tar.gz" ;;
  Linux-aarch64|Linux-arm64) asset="hugo_extended_${version}_linux-arm64.tar.gz" ;;
  *) echo "Unsupported platform: $(uname -s)-$(uname -m)" >&2; exit 1 ;;
esac

dir="$root/.cache/hugo/$version"
bin="$dir/hugo"
if [ ! -x "$bin" ]; then
  mkdir -p "$dir"
  base="https://github.com/gohugoio/hugo/releases/download/v$version"
  echo "Downloading Hugo $version ($asset)..." >&2
  curl -fsSL -o "$dir/$asset" "$base/$asset"
  curl -fsSL -o "$dir/checksums.txt" "$base/hugo_${version}_checksums.txt"
  expected="$(awk -v a="$asset" '$2 == a {print $1}' "$dir/checksums.txt")"
  if command -v shasum >/dev/null; then actual="$(shasum -a 256 "$dir/$asset")"; else actual="$(sha256sum "$dir/$asset")"; fi
  actual="${actual%% *}"
  if [ -z "$expected" ] || [ "$expected" != "$actual" ]; then
    echo "Checksum mismatch for $asset; not installing." >&2
    rm -f "$dir/$asset" "$dir/checksums.txt"
    exit 1
  fi
  tar -xzf "$dir/$asset" -C "$dir" hugo
  rm -f "$dir/$asset" "$dir/checksums.txt"
fi

cd "$root"
exec "$bin" "$@"
