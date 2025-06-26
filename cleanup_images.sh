#!/bin/bash

# Cleanup script to remove generated images

TARGET_DIR="/mnt/data/maund/open_med_flamingo/open_flamingo/data/images"

echo "🧹 Cleaning up generated images..."
echo "Directory: $TARGET_DIR"

cd "$TARGET_DIR"

# Count existing files
EXISTING=$(ls {1..10000}.jpg 2>/dev/null | wc -l)
echo "Found $EXISTING generated images to remove"

if [ $EXISTING -eq 0 ]; then
    echo "✅ No generated images found"
    exit 0
fi

# Ask for confirmation
echo ""
read -p "🤔 Remove $EXISTING generated images? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Cleanup cancelled"
    exit 0
fi

# Remove files
echo "🗑️ Removing generated images..."
rm -f {1..10000}.jpg

echo "✅ Cleanup complete!"
echo "📊 Remaining files:"
ls -la *.jpg 2>/dev/null | head -5
